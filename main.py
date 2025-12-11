import cv2
import time
import numpy as np
import os
import sys
import glob
import shutil
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
import face_recognition
import logging

# Suppress FFmpeg/libavformat warnings
logging.getLogger('opencv-python').setLevel(logging.ERROR)
os.environ['FFREPORT'] = 'file=/dev/null'
# Added 'session' to imports
from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, send_file, send_from_directory, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from functools import wraps
import threading
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta

# --- CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_security.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- CONSTANTS ---
MODEL_PATH = 'last.pt'  # Primary Fire/Smoke
MODEL_PATH_OBJECT = 'yolov8n.pt'  # Secondary General Object
BASE_RECORDINGS_DIR = "users_data"
OVERLAP_PIXELS = 44
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# --- GLOBALS ---
# We keep one pool for the whole server to manage CPU usage
detection_pool = None
yolo_model = None
yolo_model_object = None
ALL_YOLO_CLASS_NAMES = []

# Active streams: { user_id: { camera_id: VideoStreamManager } }
active_user_streams = {}
stream_lock = threading.Lock()

# Face Cache: { user_id: (encodings_list, names_list) }
face_cache = {}

# =============================================================================
# --- I. DATABASE MODELS
# =============================================================================

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    
    # Relationships
    cameras = db.relationship('Camera', backref='owner', lazy=True)
    settings = db.relationship('Settings', backref='owner', uselist=False, lazy=True)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), default="My Camera")
    source = db.Column(db.String(500), nullable=False) # Can be '0' or 'rtsp://...'
    is_active = db.Column(db.Boolean, default=True)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Detection
    yolo_enabled = db.Column(db.Boolean, default=True)
    yolo_object_enabled = db.Column(db.Boolean, default=True)
    face_recognition_enabled = db.Column(db.Boolean, default=True)
    confidence_threshold = db.Column(db.Float, default=0.4)
    object_detection_confidence = db.Column(db.Float, default=0.5)
    face_recognition_confidence = db.Column(db.Float, default=0.6)
    active_classes = db.Column(db.String(500), default="fire,smoke") # Comma separated
    
    # Email
    email_alerts_enabled = db.Column(db.Boolean, default=False)
    smtp_server = db.Column(db.String(100), default="smtp.gmail.com")
    smtp_port = db.Column(db.Integer, default=587)
    sender_email = db.Column(db.String(150))
    sender_password = db.Column(db.String(150))
    recipient_email = db.Column(db.String(150))

    # Performance
    scale_down_amount = db.Column(db.Integer, default=2)
    frame_process_interval = db.Column(db.Integer, default=3)

class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    source_name = db.Column(db.String(100))
    event_type = db.Column(db.String(50))
    description = db.Column(db.String(500))

# =============================================================================
# --- II. CORE LOGIC (Refactored for Multi-User)
# =============================================================================

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_user_face_data(user_id):
    """Loads known faces for a specific user from disk or cache."""
    global face_cache
    if user_id in face_cache:
        return face_cache[user_id]
    
    user_faces_dir = os.path.join(BASE_RECORDINGS_DIR, str(user_id), "known_faces")
    known_encodings = []
    known_names = []
    
    if os.path.exists(user_faces_dir):
        # print(f"Loading faces for User {user_id}...") 
        for name in os.listdir(user_faces_dir):
            person_dir = os.path.join(user_faces_dir, name)
            if os.path.isdir(person_dir):
                for filename in glob.glob(os.path.join(person_dir, '*.*')):
                    try:
                        image = face_recognition.load_image_file(filename)
                        encodings = face_recognition.face_encodings(image)
                        if encodings:
                            known_encodings.append(encodings[0])
                            known_names.append(name)
                    except Exception as e:
                        print(f"Error loading face {filename}: {e}")
    
    face_cache[user_id] = (known_encodings, known_names)
    return known_encodings, known_names

def detect_faces_in_chunk(chunk, x_offset, y_offset, scale_factor, upsample_amount, known_encodings, known_names, face_confidence=0.6):
    """Worker function for multiprocessing."""
    if chunk is None or chunk.size == 0:
        return []
    
    # Resize for speed
    small_chunk = cv2.resize(chunk, (0, 0), fx=scale_factor, fy=scale_factor)
    rgb_small_chunk = cv2.cvtColor(small_chunk, cv2.COLOR_BGR2RGB)
    
    chunk_face_locations = face_recognition.face_locations(rgb_small_chunk, model="hog", number_of_times_to_upsample=upsample_amount)
    chunk_face_encodings = face_recognition.face_encodings(rgb_small_chunk, chunk_face_locations, num_jitters=0)
    
    results = []
    scale_up = 1.0 / scale_factor
    # Convert confidence (0.1-1.0) to distance threshold (0.0-0.9)
    # Higher confidence = lower tolerance = stricter matching
    tolerance = 1.0 - face_confidence
    
    for (top, right, bottom, left), face_encoding in zip(chunk_face_locations, chunk_face_encodings):
        # Scale back to original size
        t_scaled = int(top * scale_up)
        r_scaled = int(right * scale_up)
        b_scaled = int(bottom * scale_up)
        l_scaled = int(left * scale_up)
        
        global_top = t_scaled + y_offset
        global_right = r_scaled + x_offset
        global_bottom = b_scaled + y_offset
        global_left = l_scaled + x_offset
        
        name = "Unknown"
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                if best_distance <= tolerance:
                    name = known_names[best_match_index]
        
        results.append((global_top, global_right, global_bottom, global_left, name))
    return results

def send_email_alert(user_settings, subject, body, image_frame=None):
    if not user_settings or not user_settings.email_alerts_enabled:
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = user_settings.sender_email
        msg['To'] = user_settings.recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_frame is not None:
            success, encoded_image = cv2.imencode('.jpg', image_frame)
            if success:
                msg.attach(MIMEImage(encoded_image.tobytes(), name="alert.jpg"))

        with smtplib.SMTP(user_settings.smtp_server, user_settings.smtp_port) as s:
            s.starttls()
            s.login(user_settings.sender_email, user_settings.sender_password)
            s.send_message(msg)
        print(f"Email sent to {user_settings.recipient_email}")
    except Exception as e:
        print(f"Email failed: {e}")

# =============================================================================
# --- III. VIDEO MANAGER (Per User/Camera)
# =============================================================================

class VideoStreamManager:
    def __init__(self, user_id, camera_id, source, settings):
        self.user_id = user_id
        self.camera_id = camera_id
        self.video_source = source
        self.cap = None
        self.last_connection_attempt = 0
        self.connection_retry_delay = 5  # seconds
        
        # Frame buffer for low-latency streaming
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.reader_thread = None
        self.reader_thread_stop = False
        
        # Cache settings values to avoid detached instance issues
        self.settings_cache = self._cache_settings(settings)
        
        # State
        self.frame_count = 0
        self.fire_alert_active = False
        self.face_detections = []
        self.yolo_detections = []
        self.recording_writer = None
        self.is_recording = False
        
        # Load user-specific faces
        self.known_encodings, self.known_names = get_user_face_data(user_id)

    def _cache_settings(self, settings_obj):
        """Cache settings values from the SQLAlchemy object to avoid detached instance errors."""
        if settings_obj:
            return {
                'yolo_enabled': settings_obj.yolo_enabled,
                'yolo_object_enabled': settings_obj.yolo_object_enabled,
                'face_recognition_enabled': settings_obj.face_recognition_enabled,
                'confidence_threshold': settings_obj.confidence_threshold,
                'object_detection_confidence': getattr(settings_obj, 'object_detection_confidence', 0.5),
                'face_recognition_confidence': getattr(settings_obj, 'face_recognition_confidence', 0.6),
                'active_classes': settings_obj.active_classes,
                'email_alerts_enabled': settings_obj.email_alerts_enabled,
                'smtp_server': settings_obj.smtp_server,
                'smtp_port': settings_obj.smtp_port,
                'sender_email': settings_obj.sender_email,
                'sender_password': settings_obj.sender_password,
                'recipient_email': settings_obj.recipient_email,
                'scale_down_amount': settings_obj.scale_down_amount,
                'frame_process_interval': settings_obj.frame_process_interval,
            }
        return {}

    def update_settings(self, settings_obj):
        """Update the settings cache with new values."""
        self.settings_cache = self._cache_settings(settings_obj)

    def _open_stream(self):
        """Lazily open the video stream with error handling."""
        try:
            self.cap = cv2.VideoCapture(self.video_source)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                # Start background frame reader thread
                self.reader_thread_stop = False
                self.reader_thread = threading.Thread(target=self._read_frames_worker, daemon=True)
                self.reader_thread.start()
                return True
            else:
                self.cap = None
                return False
        except Exception as e:
            print(f"Error opening stream {self.video_source}: {e}")
            self.cap = None
            return False

    def _read_frames_worker(self):
        """Background thread that continuously reads frames from the camera."""
        while not self.reader_thread_stop and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.current_frame = frame
            else:
                time.sleep(0.01)

    def process_frame(self):
        # Lazy load stream on first access
        if self.cap is None:
            current_time = time.time()
            if current_time - self.last_connection_attempt > self.connection_retry_delay:
                self.last_connection_attempt = current_time
                self._open_stream()
            if self.cap is None:
                return None
        
        if not self.cap.isOpened():
            self.cap = None
            return None
        
        # Get the latest frame from the buffer
        with self.frame_lock:
            frame = self.current_frame
        
        if frame is None:
            return None
        
        ret = True
        if not ret:
            # Loop video if file, or reconnect if stream
            if isinstance(self.video_source, str) and not self.video_source.startswith('rtsp'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None

        annotated_frame = frame.copy()
        
        # Settings aliases
        scale_down = self.settings_cache.get('scale_down_amount', 2)
        process_interval = self.settings_cache.get('frame_process_interval', 3)
        active_classes = self.settings_cache.get('active_classes', 'fire,smoke').split(',')
        
        # 1. Face Recognition
        # RELOAD FACES CHECK: If cache was cleared by add/delete route, reload now
        global face_cache
        if self.user_id not in face_cache:
             self.known_encodings, self.known_names = get_user_face_data(self.user_id)

        if self.settings_cache.get('face_recognition_enabled', True) and self.frame_count % process_interval == 0:
            # We use the global pool but pass USER SPECIFIC encodings
            scale_factor = 1.0 / scale_down
            h, w, _ = frame.shape
            face_conf = self.settings_cache.get('face_recognition_confidence', 0.6)
            
            # Simple single chunk for clarity in this snippet, can be split like original
            args = (frame, 0, 0, scale_factor, 1, self.known_encodings, self.known_names, face_conf)
            
            try:
                # Use apply_async to not block, or standard call. 
                # For simplicity in this structure, we do a direct call or assume pool mapping
                # Re-using the logic from original:
                results = detect_faces_in_chunk(*args)
                self.face_detections = []
                for t, r, b, l, name in results:
                    self.face_detections.append(((t, r, b, l), name))
                    
                    # Log Unknowns or Specific people
                    if name == "Unknown":
                        self.log_db_event("UNKNOWN FACE", "Unidentified person detected")
                    else:
                        self.log_db_event("ACCESS GRANTED", f"Identified {name}")
                        
            except Exception as e:
                print(f"Face Rec Error: {e}")

        # 2. YOLO
        critical_detected = False
        detected_crit_names = []
        
        self.yolo_detections = []
        
        # Primary Model
        if self.settings_cache.get('yolo_enabled', True) and yolo_model:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            results = yolo_model.predict(frame, conf=obj_conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    name = r.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    
                    is_crit = name in active_classes
                    if is_crit:
                        critical_detected = True
                        detected_crit_names.append(name)
                        
                    self.yolo_detections.append((x1, y1, x2, y2, name, is_crit))

        # Secondary Model
        if self.settings_cache.get('yolo_object_enabled', True) and yolo_model_object:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            results = yolo_model_object.predict(frame, conf=obj_conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    name = r.names.get(int(box.cls[0].item()))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    self.yolo_detections.append((x1, y1, x2, y2, name, False))

        # 3. Drawing
        for (x1, y1, x2, y2, name, is_crit) in self.yolo_detections:
            color = (0, 0, 255) if is_crit else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        for (top, right, bottom, left), name in self.face_detections:
            color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
            cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
            cv2.putText(annotated_frame, name, (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 4. Alerts
        if critical_detected and not self.fire_alert_active:
            msg = f"Detected: {', '.join(set(detected_crit_names))}"
            self.log_db_event("CRITICAL ALERT", msg)
            # Create a simple settings object for email alert
            class SettingsObj:
                pass
            settings_obj = SettingsObj()
            for k, v in self.settings_cache.items():
                setattr(settings_obj, k, v)
            threading.Thread(target=send_email_alert, args=(settings_obj, "CRITICAL ALERT", msg, annotated_frame)).start()
        
        self.fire_alert_active = critical_detected
        
        # 5. Recording
        if self.is_recording:
            if not self.recording_writer:
                self.start_recording(annotated_frame)
            self.recording_writer.write(annotated_frame)
        elif self.recording_writer:
            self.stop_recording()

        self.frame_count += 1
        return annotated_frame

    def start_recording(self, frame):
        user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(self.user_id), "recordings")
        os.makedirs(user_rec_dir, exist_ok=True)
        filename = os.path.join(user_rec_dir, f"cam_{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.webm")
        h, w, _ = frame.shape
        self.recording_writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'VP80'), 15.0, (w, h))

    def stop_recording(self):
        if self.recording_writer:
            self.recording_writer.release()
            self.recording_writer = None

    def log_db_event(self, event_type, desc):
        # We need an app context to write to DB from a thread
        with app.app_context():
            log = EventLog(user_id=self.user_id, source_name=f"Cam {self.camera_id}", event_type=event_type, description=desc)
            db.session.add(log)
            db.session.commit()

    def release(self):
        self.stop_recording()
        self.reader_thread_stop = True
        if self.reader_thread:
            self.reader_thread.join(timeout=2)
        if self.cap:
            self.cap.release()

# =============================================================================
# --- IV. ROUTES
# =============================================================================

# --- PRIVACY DISCLAIMER GATEKEEPER ---

@app.before_request
def check_privacy_agreement():
    """
    This function runs before EVERY request. 
    It checks if the user has accepted the disclaimer in the current session.
    """
    # 1. Define routes that MUST be accessible without the agreement (static files, the disclaimer itself, etc)
    allowed_endpoints = ['disclaimer', 'accept_terms', 'static']
    
    # 2. Check if the requested endpoint is allowed
    if request.endpoint in allowed_endpoints:
        return

    # 3. Check if 'privacy_agreed' is in the session
    if not session.get('privacy_agreed'):
        # If not, force redirect to the disclaimer page
        return redirect(url_for('disclaimer'))

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/accept_terms', methods=['POST'])
def accept_terms():
    # Set session variable to indicate agreement
    session['privacy_agreed'] = True
    # Redirect to login (or index if they were already logged in but session expired)
    return redirect(url_for('login'))

# --- END GATEKEEPER ---


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))
            
        new_user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(new_user)
        db.session.commit()
        
        # Initialize default settings
        default_settings = Settings(user_id=new_user.id, recipient_email=email)
        db.session.add(default_settings)
        db.session.commit()
        
        # Create directories
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "known_faces"), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "recordings"), exist_ok=True)
        
        login_user(new_user)
        return redirect(url_for('index'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Login failed. Check details.', 'danger')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    # Stop streams for this user to save resources
    with stream_lock:
        if current_user.id in active_user_streams:
            for mgr in active_user_streams[current_user.id].values():
                mgr.release()
            del active_user_streams[current_user.id]
            
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    user_cameras = Camera.query.filter_by(user_id=current_user.id).all()
    # If first time, active_user_streams might be empty.
    # We don't start streams here; we start them when the image is requested.
    return render_template('index.html', cameras=user_cameras)

@app.route('/add_camera', methods=['POST'])
@login_required
def add_camera():
    source = request.form.get('source') # URL
    name = request.form.get('name')
    
    # 1. Reject integers (Local Webcams)
    if source.isdigit():
        flash('Local Device IDs (0, 1, etc.) are not supported in Cloud Mode. Please provide an RTSP/HTTP URL.', 'danger')
        return redirect(url_for('index'))
    
    if source:
        new_cam = Camera(user_id=current_user.id, source=source, name=name)
        db.session.add(new_cam)
        db.session.commit()
        flash('Camera added.', 'success')
        
    return redirect(url_for('index'))

@app.route('/delete_camera/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    
    # Ensure ownership
    if camera.user_id != current_user.id:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('index'))
    
    # 1. Stop active stream if running
    with stream_lock:
        if current_user.id in active_user_streams:
            if camera.id in active_user_streams[current_user.id]:
                manager = active_user_streams[current_user.id][camera.id]
                manager.release()
                del active_user_streams[current_user.id][camera.id]
    
    # 2. Remove from DB
    try:
        db.session.delete(camera)
        db.session.commit()
        flash(f'Camera "{camera.name}" removed.', 'success')
    except Exception as e:
        flash(f'Error deleting camera: {e}', 'danger')
        
    return redirect(url_for('index'))

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    # Security check: Ensure camera belongs to user
    camera = Camera.query.get_or_404(camera_id)
    if camera.user_id != current_user.id:
        return "Unauthorized", 403

    return Response(gen_frames(current_user.id, camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(user_id, camera):
    """Generator that manages the VideoStreamManager for a specific user/camera."""
    global active_user_streams
    
    manager = None
    with stream_lock:
        if user_id not in active_user_streams:
            active_user_streams[user_id] = {}
            
        if camera.id not in active_user_streams[user_id]:
            # Load User Settings
            with app.app_context():
                user_settings = Settings.query.filter_by(user_id=user_id).first()
            manager = VideoStreamManager(user_id, camera.id, camera.source, user_settings)
            active_user_streams[user_id][camera.id] = manager
        else:
            manager = active_user_streams[user_id][camera.id]
            
    while True:
        frame = manager.process_frame()
        if frame is not None:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            time.sleep(0.01)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_settings = Settings.query.filter_by(user_id=current_user.id).first()
    
    # List existing faces
    user_faces_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces")
    known_faces_list = []
    if os.path.exists(user_faces_dir):
        known_faces_list = [name for name in os.listdir(user_faces_dir) if os.path.isdir(os.path.join(user_faces_dir, name))]

    if request.method == 'POST':
        # Update settings
        user_settings.yolo_enabled = 'yolo_enabled' in request.form
        user_settings.face_recognition_enabled = 'face_rec_enabled' in request.form
        user_settings.email_alerts_enabled = 'email_enabled' in request.form
        
        user_settings.confidence_threshold = float(request.form.get('confidence', 0.4))
        user_settings.object_detection_confidence = float(request.form.get('object_detection_confidence', 0.5))
        user_settings.face_recognition_confidence = float(request.form.get('face_recognition_confidence', 0.6))
        user_settings.smtp_server = request.form.get('smtp_server')
        user_settings.sender_email = request.form.get('sender_email')
        user_settings.sender_password = request.form.get('sender_password')
        user_settings.recipient_email = request.form.get('recipient_email')
        user_settings.smtp_port = int(request.form.get('smtp_port', 587)) # Capture port correctly
        
        db.session.commit()
        flash("Settings Updated", "success")
        
        # Reload stream manager settings if active
        with stream_lock:
            if current_user.id in active_user_streams:
                for mgr in active_user_streams[current_user.id].values():
                    mgr.update_settings(user_settings)
        
        return redirect(url_for('settings'))
        
    return render_template('settings.html', settings=user_settings, known_faces=known_faces_list)

@app.route('/add_face', methods=['POST'])
@login_required
def add_face():
    if 'file' not in request.files:
        flash('No file part', 'danger')
        return redirect(url_for('settings'))
    
    file = request.files['file']
    name = request.form.get('name')
    
    if file.filename == '' or not name:
        flash('No selected file or name missing', 'danger')
        return redirect(url_for('settings'))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Sanitize name to prevent directory traversal
        safe_name = secure_filename(name)
        
        save_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces", safe_name)
        os.makedirs(save_dir, exist_ok=True)
        
        file.save(os.path.join(save_dir, filename))
        
        # Invalidate cache so it reloads on next frame
        global face_cache
        if current_user.id in face_cache:
            del face_cache[current_user.id]
            
        flash(f'Face for "{safe_name}" added successfully.', 'success')
    else:
        flash('Invalid file type. Allowed: png, jpg, jpeg', 'danger')
        
    return redirect(url_for('settings'))

@app.route('/delete_face/<name>', methods=['POST'])
@login_required
def delete_face(name):
    safe_name = secure_filename(name)
    face_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces", safe_name)
    
    if os.path.exists(face_dir):
        try:
            shutil.rmtree(face_dir)
            
            # Invalidate cache
            global face_cache
            if current_user.id in face_cache:
                del face_cache[current_user.id]
                
            flash(f'Face "{safe_name}" deleted.', 'success')
        except Exception as e:
            flash(f'Error deleting face: {e}', 'danger')
    else:
        flash('Face not found.', 'danger')
        
    return redirect(url_for('settings'))


@app.route('/toggle_recording/<int:camera_id>', methods=['POST'])
@login_required
def toggle_recording(camera_id):
    # Find manager
    with stream_lock:
        if current_user.id in active_user_streams and camera_id in active_user_streams[current_user.id]:
            mgr = active_user_streams[current_user.id][camera_id]
            mgr.is_recording = not mgr.is_recording
            status = "Started" if mgr.is_recording else "Stopped"
            return jsonify({"success": True, "message": f"Recording {status}"})
    return jsonify({"error": "Camera not active"}), 400

@app.route('/recordings')
@login_required
def recordings_page():
    return render_template('recordings.html')

@app.route('/events')
@login_required
def events_page():
    return render_template('events.html')

@app.route('/api/recordings', methods=['GET'])
@login_required
def api_recordings():
    user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "recordings")
    recordings = []
    
    if os.path.exists(user_rec_dir):
        for filename in sorted(os.listdir(user_rec_dir), reverse=True):
            if filename.endswith('.webm'):
                # Parse filename: cam_<camera_id>_<YYYYMMDD_HHMMSS>.webm
                parts = filename.replace('.webm', '').split('_')
                if len(parts) >= 4:
                    try:
                        cam_id = parts[1]
                        date_str = parts[2]
                        time_str = parts[3]
                        recordings.append({
                            'filename': filename,
                            'cam': cam_id,
                            'date': date_str,
                            'time': time_str
                        })
                    except Exception as e:
                        print(f"Error parsing recording {filename}: {e}")
    
    return jsonify(recordings)

@app.route('/api/events', methods=['GET'])
@login_required
def api_events():
    events = EventLog.query.filter_by(user_id=current_user.id).order_by(EventLog.timestamp.desc()).all()
    result = []
    for event in events:
        result.append({
            'id': event.id,
            'timestamp': event.timestamp.isoformat(),
            'source_name': event.source_name,
            'event_type': event.event_type,
            'description': event.description
        })
    return jsonify(result)

@app.route('/recordings/<filename>')
@login_required
def get_recording(filename):
    user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "recordings")
    filepath = os.path.join(user_rec_dir, filename)
    
    # Security check: ensure file is in user's directory and prevent directory traversal
    filepath_abs = os.path.abspath(filepath)
    user_rec_dir_abs = os.path.abspath(user_rec_dir)
    if not filepath_abs.startswith(user_rec_dir_abs):
        return "Unauthorized", 403
    
    if os.path.exists(filepath):
        return send_file(filepath, mimetype='video/webm', as_attachment=True, download_name=filename)
    return "File not found", 404

@app.route('/api/delete_recording/<filename>', methods=['POST'])
@login_required
def delete_recording(filename):
    user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "recordings")
    filepath = os.path.join(user_rec_dir, filename)
    
    # Security check: ensure file is in user's directory and prevent directory traversal
    filepath_abs = os.path.abspath(filepath)
    user_rec_dir_abs = os.path.abspath(user_rec_dir)
    if not filepath_abs.startswith(user_rec_dir_abs):
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({"success": True, "message": "Recording deleted"})
        return jsonify({"error": "File not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_all_recordings', methods=['POST'])
@login_required
def clear_all_recordings():
    user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "recordings")
    
    try:
        if os.path.exists(user_rec_dir):
            for filename in os.listdir(user_rec_dir):
                filepath = os.path.join(user_rec_dir, filename)
                if os.path.isfile(filepath) and filename.endswith('.webm'):
                    os.remove(filepath)
        return jsonify({"success": True, "message": "All recordings cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/delete_event/<int:event_id>', methods=['POST'])
@login_required
def delete_event(event_id):
    event = EventLog.query.get_or_404(event_id)
    
    # Security check: ensure event belongs to user
    if event.user_id != current_user.id:
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        db.session.delete(event)
        db.session.commit()
        return jsonify({"success": True, "message": "Event deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/clear_all_events', methods=['POST'])
@login_required
def clear_all_events():
    try:
        EventLog.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"success": True, "message": "All events cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- INITIALIZATION ---
def init_app():
    global yolo_model, yolo_model_object, detection_pool, ALL_YOLO_CLASS_NAMES
    
    with app.app_context():
        db.create_all()
        
    yolo_model = YOLO(MODEL_PATH)
    if yolo_model.names:
        ALL_YOLO_CLASS_NAMES = list(yolo_model.names.values())
        
    yolo_model_object = YOLO(MODEL_PATH_OBJECT)
    
    detection_pool = Pool(processes=cpu_count())
    print("App Initialized.")

if __name__ == '__main__':
    init_app()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)

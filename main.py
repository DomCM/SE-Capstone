import argparse
import cv2
import time
import numpy as np
import os
import sys
import glob
import shutil
import queue
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
import face_recognition
import logging
import click

# Try to import torch to detect NVIDIA CUDA capabilities
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        CUDA_DEVICE_NAME = torch.cuda.get_device_name(0)
    else:
        CUDA_DEVICE_NAME = "None"
except ImportError:
    CUDA_AVAILABLE = False
    CUDA_DEVICE_NAME = "None"

# ══════════════════════════════════════════════════════════════
# OPTIMIZATION SETTINGS
# ══════════════════════════════════════════════════════════════
# Limit OpenVINO to 2 CPU threads when CPU is used as backup to avoid starving Flask
os.environ['OV_CPU_THREADS_NUM'] = '2'
# Enable model caching to decrease compilation delay on subsequent app launches
os.environ['OPENVINO_CACHE_DIR'] = 'ov_cache'

logging.getLogger('opencv-python').setLevel(logging.ERROR)
os.environ['FFREPORT'] = 'file=/dev/null'

from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, send_file, send_from_directory, flash, session
from flask_sqlalchemy import SQLAlchemy
from io import BytesIO
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
from sqlalchemy import inspect, text

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_security.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Dynamic Model Configuration paths
# We will resolve these to PyTorch (.pt), TensorRT (.engine), or OpenVINO directories depending on hardware
MODEL_PATH = 'yolo26n.pt'                # Primary Model (Fallback to 'last_openvino_model' if only OpenVINO is available)
MODEL_PATH_OBJECT = 'best.pt'      # Secondary Model (Fallback to 'yolov8n_openvino_model' if only OpenVINO is available)

BASE_RECORDINGS_DIR = "users_data"
OVERLAP_PIXELS = 44
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

detection_pool = None
yolo_model = None
yolo_model_object = None
ALL_YOLO_CLASS_NAMES = []
ACCELERATOR_DEVICE = "cpu"  # Will be resolved dynamically to 'cuda' (NVIDIA GPU), 'GPU' (Intel), or 'cpu'

active_user_streams = {}
stream_lock = threading.Lock()

face_cache = {}

# ══════════════════════════════════════════════════════════════
# ASYNCHRONOUS DATABASE LOGGING QUEUE
# ══════════════════════════════════════════════════════════════
db_log_queue = queue.Queue()


def is_human_detection_name(name):
    """Return True for common YOLO labels that represent a person."""
    if not name:
        return False
    normalized = str(name).strip().lower()
    return normalized in {"person", "people", "persons", "human", "humans"}


def calculate_overlap_ratio(box_a, box_b):
    """Compute intersection-over-union for two bounding boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection_area = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = max(1, area_a + area_b - intersection_area)
    return intersection_area / union_area


def should_run_face_recognition(frame_count, process_interval, has_active_tracker, tracker_lost):
    """Decide when to do full face recognition versus KCF tracking."""
    if frame_count == 0:
        return True
    if tracker_lost:
        return True
    if process_interval <= 0:
        return True
    return frame_count % process_interval == 0 or not has_active_tracker


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    
    #relationships
    cameras = db.relationship('Camera', backref='owner', lazy=True)
    settings = db.relationship('Settings', backref='owner', uselist=False, lazy=True)

class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), default="My Camera")
    source = db.Column(db.String(500), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_public = db.Column(db.Boolean, default=False)

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    #scan
    yolo_enabled = db.Column(db.Boolean, default=True)
    yolo_object_enabled = db.Column(db.Boolean, default=True)
    face_recognition_enabled = db.Column(db.Boolean, default=True)
    confidence_threshold = db.Column(db.Float, default=0.4)
    object_detection_confidence = db.Column(db.Float, default=0.5)
    face_recognition_confidence = db.Column(db.Float, default=0.6)
    active_classes = db.Column(db.String(500), default="fire,smoke")

    # ACCESS CONTROL SETTINGS
    allow_registration = db.Column(db.Boolean, default=True)
    require_disclaimer = db.Column(db.Boolean, default=False)
    session_timeout_enabled = db.Column(db.Boolean, default=False)
    session_timeout_minutes = db.Column(db.Integer, default=60)
    
    #email
    email_alerts_enabled = db.Column(db.Boolean, default=False)
    smtp_server = db.Column(db.String(100), default="smtp.gmail.com")
    smtp_port = db.Column(db.Integer, default=587)
    sender_email = db.Column(db.String(150))
    sender_password = db.Column(db.String(150))
    recipient_email = db.Column(db.String(150))

    #perf
    scale_down_amount = db.Column(db.Integer, default=2)
    frame_process_interval = db.Column(db.Integer, default=3)

class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    source_name = db.Column(db.String(100))
    event_type = db.Column(db.String(50))
    description = db.Column(db.String(500))

    user = db.relationship('User', backref='event_logs', lazy=True)

    @property
    def username(self):
        if self.user:
            return self.user.username or self.user.email or 'Unknown User'
        return 'Unknown User'

    @property
    def action_type(self):
        return (self.event_type or 'other').lower()

    @property
    def action_label(self):
        mapping = {
            'login': 'Login',
            'logout': 'Logout',
            'create': 'Create',
            'delete': 'Delete',
            'settings': 'Settings',
            'motion': 'Motion',
            'alert': 'Alert',
            'recording': 'Recording',
        }
        return mapping.get(self.action_type, self.action_type.title())

    @property
    def detail(self):
        return self.description or self.source_name or '—'

    @property
    def ip_address(self):
        return None


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

ADMIN_EMAILS = [email.strip().lower() for email in os.environ.get('ADMIN_EMAILS', '').split(',') if email.strip()]


def get_user_role(user):
    if user is None:
        return 'user'
    role = getattr(user, 'role', None)
    if role is None:
        return 'user'
    return str(role).strip().lower() or 'user'


def is_admin_user(user_or_email):
    if user_or_email is None:
        return False
    if isinstance(user_or_email, str):
        email = user_or_email.strip().lower()
        if email in ADMIN_EMAILS:
            return True
        user = User.query.filter_by(email=email).first()
        return bool(user and get_user_role(user) == 'admin')
    if hasattr(user_or_email, 'email'):
        email = getattr(user_or_email, 'email', '').strip().lower()
        if email in ADMIN_EMAILS:
            return True
    return get_user_role(user_or_email) == 'admin'


def get_dashboard_target(user):
    return 'admin_dashboard' if is_admin_user(user) else 'index'


def ensure_database_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        user_columns = {column['name'] for column in inspector.get_columns('user')}
        if 'role' not in user_columns:
            db.session.execute(text("ALTER TABLE user ADD COLUMN role VARCHAR(20) NOT NULL DEFAULT 'user'"))
            db.session.commit()

        camera_columns = {column['name'] for column in inspector.get_columns('camera')}
        if 'is_public' not in camera_columns:
            db.session.execute(text("ALTER TABLE camera ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0"))
            db.session.commit()


def create_admin_user(username, email, password):
    if not username or not email or not password:
        raise ValueError('Username, email, and password are required.')

    with app.app_context():
        ensure_database_schema()
        existing = User.query.filter((User.email == email) | (User.username == username)).first()
        if existing:
            raise ValueError('An account with that username or email already exists.')

        new_user = User(username=username, email=email, password=generate_password_hash(password), role='admin')
        db.session.add(new_user)
        db.session.commit()

        default_settings = Settings(user_id=new_user.id, recipient_email=email)
        db.session.add(default_settings)
        db.session.commit()

        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'known_faces'), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'recordings'), exist_ok=True)
        return new_user


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

def detect_faces_in_chunk(full_frame, bbox, scale_factor, upsample_amount, known_encodings, known_names, face_confidence=0.6):
    """
    Optimized worker function. Processes facial recognition only within a specific bounding box (ROI).
    """
    if full_frame is None or bbox is None:
        return []

    x1, y1, x2, y2 = map(int, bbox)
    cropped_image = full_frame[y1:y2, x1:x2]
    
    if cropped_image.size == 0:
        return []
        
    if scale_factor != 1.0:
        cropped_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor)
        
    rgb_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    
    chunk_face_locations = face_recognition.face_locations(rgb_cropped_image, model="hog", number_of_times_to_upsample=upsample_amount)
    chunk_face_encodings = face_recognition.face_encodings(rgb_cropped_image, chunk_face_locations, num_jitters=0)
    
    results = []
    scale_up = 1.0 / scale_factor
    tolerance = 1.0 - face_confidence
    
    for (top, right, bottom, left), face_encoding in zip(chunk_face_locations, chunk_face_encodings):
        t_scaled = int((top * scale_up) + y1)
        r_scaled = int((right * scale_up) + x1)
        b_scaled = int((bottom * scale_up) + y1)
        l_scaled = int((left * scale_up) + x1)
        
        name = "Unknown"
        if known_encodings:
            face_distances = face_recognition.face_distance(known_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_distance = face_distances[best_match_index]
                if best_distance <= tolerance:
                    name = known_names[best_match_index]
        
        results.append((t_scaled, r_scaled, b_scaled, l_scaled, name))
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


# ══════════════════════════════════════════════════════════════
# BACKGROUND EVENT LOGGER THREAD (PREVENTS SQLITE THREAD LOCKING)
# ══════════════════════════════════════════════════════════════
def db_logger_worker():
    """Consumes event logs asynchronously from a queue to prevent blocking streaming pipelines."""
    while True:
        try:
            log_data = db_log_queue.get()
            if log_data is None:
                break
            
            with app.app_context():
                log = EventLog(
                    user_id=log_data['user_id'],
                    source_name=log_data['source_name'],
                    event_type=log_data['event_type'],
                    description=log_data['description']
                )
                db.session.add(log)
                db.session.commit()
            db_log_queue.task_done()
        except Exception as e:
            print(f"[DB LOGGER ERROR] Failed to write event log: {e}")
            time.sleep(0.1)


class VideoStreamManager:
    def __init__(self, user_id, camera_id, source, settings):
        self.user_id = user_id
        self.camera_id = camera_id
        self.video_source = source
        self.cap = None
        self.last_connection_attempt = 0
        self.connection_retry_delay = 5
        
        self.current_frame = None
        self.frame_lock = threading.Lock()
        
        # Thread management
        self.reader_thread = None
        self.reader_thread_stop = False
        self.processor_thread = None
        self.processor_thread_stop = False
        
        # Memory buffer for background processed & compressed JPEG bytes
        self.latest_annotated_frame = None
        self.latest_frame_lock = threading.Lock()
        
        self.settings_cache = self._cache_settings(settings)
        
        self.frame_count = 0
        self.fire_alert_active = False
        self.face_detections = []
        self.yolo_detections = []
        self.recording_writer = None
        self.is_recording = False
        
        self.trackers = []          
        self.tracker_active = False
        self.tracker_lost = False
        self.tracking_scale = 1.0
        
        self.known_encodings, self.known_names = get_user_face_data(user_id)
        self.debug_tracker = True

        # Spawns persistent workers immediately to connect & handle self-healing on connection loss
        self.reader_thread_stop = False
        self.reader_thread = threading.Thread(target=self._read_frames_worker, daemon=True)
        self.reader_thread.start()
        
        self.processor_thread_stop = False
        self.processor_thread = threading.Thread(target=self._process_frames_worker, daemon=True)
        self.processor_thread.start()

    def _cache_settings(self, settings_obj):
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
        self.settings_cache = self._cache_settings(settings_obj)

    def _reset_tracker(self):
        if self.debug_tracker and self.tracker_active:
            print(f"[TRACKERS] Resetting/Clearing trackers at frame {self.frame_count}")
        self.trackers = []
        self.tracker_active = False
        self.tracker_lost = True

    def _initialize_trackers(self, frame, bboxes):
        self.trackers = []
        if frame is None or not bboxes:
            self.tracker_active = False
            self.tracker_lost = True
            return

        # Target a responsive resolution (e.g., width of 480px) for Fast Fourier transforms inside KCF
        h, w = frame.shape[:2]
        target_w = 480
        self.tracking_scale = min(1.0, target_w / w) if w > target_w else 1.0

        if self.tracking_scale < 1.0:
            tracking_frame = cv2.resize(frame, (0, 0), fx=self.tracking_scale, fy=self.tracking_scale)
        else:
            tracking_frame = frame

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Downscale coordinate bounds matching the downscaled tracking_frame
            sx1 = int(x1 * self.tracking_scale)
            sy1 = int(y1 * self.tracking_scale)
            sx2 = int(x2 * self.tracking_scale)
            sy2 = int(y2 * self.tracking_scale)
            
            width = max(1, sx2 - sx1)
            height = max(1, sy2 - sy1)
            try:
                try:
                    tracker = cv2.legacy.TrackerKCF.create()
                    tracker_name = "KCF (legacy)"
                except (AttributeError, NameError):
                    tracker = cv2.TrackerKCF.create()
                    tracker_name = "KCF"
                    
                ok = tracker.init(tracking_frame, (sx1, sy1, width, height))
                if ok:
                    self.trackers.append({
                        "tracker": tracker,
                        "bbox": (sx1, sy1, sx2, sy2)
                    })
                    if self.debug_tracker:
                        print(f"[TRACKER] Initialized downscaled {tracker_name} at scale {self.tracking_scale:.2f}")
            except Exception as e:
                print(f"Tracker init error: {e}")

        if self.trackers:
            self.tracker_active = True
            self.tracker_lost = False
        else:
            self.tracker_active = False
            self.tracker_lost = True

    def _update_trackers(self, frame):
        if not self.trackers or frame is None:
            self.tracker_active = False
            self.tracker_lost = True
            return []

        if self.tracking_scale < 1.0:
            tracking_frame = cv2.resize(frame, (0, 0), fx=self.tracking_scale, fy=self.tracking_scale)
        else:
            tracking_frame = frame

        updated_bboxes = []
        active_trackers = []

        from concurrent.futures import ThreadPoolExecutor

        # Parallelize the individual tracker updates across multiple CPU cores (GIL-free in OpenCV C++)
        def update_single_tracker(item):
            tracker = item["tracker"]
            try:
                ok, bbox = tracker.update(tracking_frame)
                if ok:
                    sx, sy, sw, sh = [int(v) for v in bbox]
                    
                    # Convert bounding coordinates back to native resolution for overlays and recording
                    scale_inv = 1.0 / self.tracking_scale
                    rx1 = int(sx * scale_inv)
                    ry1 = int(sy * scale_inv)
                    rx2 = int((sx + sw) * scale_inv)
                    ry2 = int((sy + sh) * scale_inv)
                    
                    return {
                        "success": True,
                        "tracker_item": {
                            "tracker": tracker,
                            "bbox": (sx, sy, sx + sw, sy + sh)
                        },
                        "original_bbox": (rx1, ry1, rx2, ry2)
                    }
            except Exception as e:
                print(f"Parallel Tracker update error: {e}")
            return {"success": False}

        with ThreadPoolExecutor(max_workers=min(4, len(self.trackers))) as executor:
            results = list(executor.map(update_single_tracker, self.trackers))

        for res in results:
            if res["success"]:
                active_trackers.append(res["tracker_item"])
                updated_bboxes.append(res["original_bbox"])

        self.trackers = active_trackers
        if self.trackers:
            self.tracker_active = True
            self.tracker_lost = False
        else:
            self.tracker_active = False
            self.tracker_lost = True

        return updated_bboxes

    def _read_frames_worker(self):
        """Asynchronous stream reader. Connects, monitors, and automatically reconnects video pipelines."""
        while not self.reader_thread_stop:
            # Reconnection check & execution
            if self.cap is None or not self.cap.isOpened():
                current_time = time.time()
                if current_time - self.last_connection_attempt > self.connection_retry_delay:
                    self.last_connection_attempt = current_time
                    if self.cap:
                        try:
                            self.cap.release()
                        except Exception:
                            pass
                    try:
                        self.cap = cv2.VideoCapture(self.video_source)
                        if self.cap and self.cap.isOpened():
                            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                            print(f"[STREAM SUCCESS] Connected to camera source: {self.video_source}")
                        else:
                            self.cap = None
                            print(f"[STREAM FAILED] Couldn't connect to: {self.video_source}. Retrying...")
                    except Exception as e:
                        self.cap = None
                        print(f"[STREAM ERROR] Exception opening {self.video_source}: {e}")
                time.sleep(1.0)
                continue

            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.frame_lock:
                        self.current_frame = frame
                else:
                    # Stream disconnected or closed on remote server
                    print(f"[STREAM ERROR] Frame read returned empty. Re-triggering connection.")
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
            except Exception as e:
                print(f"[STREAM ERROR] Thread exception during reads: {e}")
                self.cap = None
            time.sleep(0.01)

    def _process_frames_worker(self):
        """Dedicated processor running asynchronously to completely decouple AI latency from Flask HTTP threads."""
        while not self.processor_thread_stop:
            with self.frame_lock:
                frame = self.current_frame

            if frame is None:
                time.sleep(0.05)
                continue

            # Run computationally intensive detection pipelines
            try:
                annotated_frame = self.do_process_frame(frame)

                if annotated_frame is not None:
                    # Pre-compress to JPEG here so Flask threads don't block compressing images
                    ret, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ret:
                        with self.latest_frame_lock:
                            self.latest_annotated_frame = buffer.tobytes()
            except Exception as e:
                print(f"[PROCESSOR ERROR] Exception in process worker: {e}")

            time.sleep(0.005)

    def get_latest_frame_bytes(self):
        """Instantly read pre-compiled frame bytes from shared memory (0ms Flask thread processing)."""
        with self.latest_frame_lock:
            return self.latest_annotated_frame

    def do_process_frame(self, frame):
        """Worker executing detection loops (formerly process_frame)."""
        annotated_frame = frame.copy()
        
        scale_down = self.settings_cache.get('scale_down_amount', 2)
        process_interval = max(1, self.settings_cache.get('frame_process_interval', 3))
        active_classes = self.settings_cache.get('active_classes', 'fire,smoke').split(',')
        
        global face_cache
        if self.user_id not in face_cache:
             self.known_encodings, self.known_names = get_user_face_data(self.user_id)

        # ══════════════════════════════════════════════════════════════
        # 1. KCF TRACKING WITH PERIODIC FACE RE-VERIFICATION
        # ══════════════════════════════════════════════════════════════
        if self.tracker_active:
            tracked_boxes = self._update_trackers(frame)
            
            if tracked_boxes:
                for x1, y1, x2, y2 in tracked_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                if self.frame_count % process_interval == 0 and self.settings_cache.get('face_recognition_enabled', True):
                    scale_factor = 1.0 / scale_down
                    face_conf = self.settings_cache.get('face_recognition_confidence', 0.6)
                    
                    try:
                        self.face_detections = []
                        for tracked_box in tracked_boxes:
                            results = detect_faces_in_chunk(frame, tracked_box, scale_factor, 1, self.known_encodings, self.known_names, face_conf)
                            for t, r, b, l, name in results:
                                self.face_detections.append(((t, r, b, l), name))
                    except Exception as e:
                        print(f"Periodic Face Rec Error: {e}")
                
                for (top, right, bottom, left), name in self.face_detections:
                    color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(annotated_frame, f"Verified: {name}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(annotated_frame, f"Tracking Active ({len(tracked_boxes)} Persons — YOLO Suspended)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            else:
                self._reset_tracker()

        # ══════════════════════════════════════════════════════════════
        # 2. SEARCH STATE (YOLO ON INTEL iGPU/CPU OR NVIDIA CUDA + FACE)
        # ══════════════════════════════════════════════════════════════
        if not self.tracker_active:
            critical_detected = False
            detected_crit_names = []
            self.yolo_detections = []
            human_boxes = []
            
            # --- Primary YOLO Model ---
            if self.settings_cache.get('yolo_enabled', True) and yolo_model:
                obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
                results = yolo_model.predict(frame, conf=obj_conf, verbose=False, device=ACCELERATOR_DEVICE)
                for r in results:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        name = r.names.get(cls_id, str(cls_id))
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                        
                        is_crit = name in active_classes
                        if is_crit:
                            critical_detected = True
                            detected_crit_names.append(name)
                        
                        self.yolo_detections.append((x1, y1, x2, y2, name, is_crit))
                        if is_human_detection_name(name):
                            human_boxes.append((x1, y1, x2, y2, confidence))

            # --- Secondary YOLO Model ---
            if self.settings_cache.get('yolo_object_enabled', True) and yolo_model_object:
                obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
                results = yolo_model_object.predict(frame, conf=obj_conf, verbose=False, device=ACCELERATOR_DEVICE)
                for r in results:
                    for box in r.boxes:
                        name = r.names.get(int(box.cls[0].item()))
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                        self.yolo_detections.append((x1, y1, x2, y2, name, False))
                        if is_human_detection_name(name):
                            human_boxes.append((x1, y1, x2, y2, confidence))

            for (x1, y1, x2, y2, name, is_crit) in self.yolo_detections:
                color = (0, 0, 255) if is_crit else (0, 255, 0)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if human_boxes and self.settings_cache.get('face_recognition_enabled', True):
                scale_factor = 1.0 / scale_down
                face_conf = self.settings_cache.get('face_recognition_confidence', 0.6)
                
                roi_tasks = []
                for bbox in human_boxes:
                    roi_tasks.append((frame, bbox[:4], scale_factor, 1, self.known_encodings, self.known_names, face_conf))

                try:
                    all_results = detection_pool.starmap(detect_faces_in_chunk, roi_tasks)
                    
                    self.face_detections = []
                    for results_list in all_results:
                        for t, r, b, l, name in results_list:
                            self.face_detections.append(((t, r, b, l), name))
                            
                            if name == "Unknown":
                                self.log_db_event("UNKNOWN FACE", "Unidentified person detected in an ROI.")
                            else:
                                self.log_db_event("RECOGNIZED", f"Identified {name}")

                except Exception as e:
                    print(f"Parallel Face Recognition Error: {e}")

                for (top, right, bottom, left), name in self.face_detections:
                    color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(annotated_frame, name, (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                bboxes_to_track = [box[:4] for box in human_boxes]
                self._initialize_trackers(frame, bboxes_to_track)

            if critical_detected and not self.fire_alert_active:
                msg = f"Detected: {', '.join(set(detected_crit_names))}"
                self.log_db_event("CRITICAL ALERT", msg)
                class SettingsObj:
                    pass
                settings_obj = SettingsObj()
                for k, v in self.settings_cache.items():
                    setattr(settings_obj, k, v)
                threading.Thread(target=send_email_alert, args=(settings_obj, "CRITICAL ALERT", msg, annotated_frame)).start()
            
            self.fire_alert_active = critical_detected

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
        """Add event logs asynchronously to the thread-safe database queue."""
        db_log_queue.put({
            'user_id': self.user_id,
            'source_name': f"Cam {self.camera_id}",
            'event_type': event_type,
            'description': desc
        })

    def release(self):
        self.stop_recording()
        self.reader_thread_stop = True
        self.processor_thread_stop = True
        if self.reader_thread:
            self.reader_thread.join(timeout=1)
        if self.processor_thread:
            self.processor_thread.join(timeout=1)
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass


def is_mobile(request):
    if 'User-Agent' not in request.headers:
        return False
    
    user_agent = request.headers['User-Agent'].lower()
    mobile_keywords = [
        'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone', 'opera mini'
    ]
    
    for keyword in mobile_keywords:
        if keyword in user_agent:
            return True
    return False


@app.before_request
def check_privacy_agreement():
    allowed_endpoints = ['disclaimer', 'accept_terms', 'static']
    if request.endpoint in allowed_endpoints:
        return
    if not session.get('privacy_agreed'):
        return redirect(url_for('disclaimer'))

@app.route('/disclaimer')
def disclaimer():
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}disclaimer.html')

@app.route('/accept_terms', methods=['POST'])
def accept_terms():
    session['privacy_agreed'] = True
    return redirect(url_for('login'))


def admin_required(view_func):
    @wraps(view_func)
    @login_required
    def wrapped(*args, **kwargs):
        if not is_admin_user(current_user):
            flash('You do not have permission to access the admin area.', 'danger')
            return redirect(url_for(get_dashboard_target(current_user)))
        return view_func(*args, **kwargs)
    return wrapped


@app.route('/register', methods=['GET', 'POST'])
def register():
    with app.app_context():
        settings = Settings.query.first()
        
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            flash('All fields are required.', 'danger')
            return redirect(url_for('register'))

        if User.query.filter_by(username=username).first():
            flash('This username is already taken.', 'danger')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))

        if settings and not settings.allow_registration:
            flash("Public registration is disabled by the administrator.", "danger")
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, password=generate_password_hash(password), role='user')
        db.session.add(new_user)
        db.session.commit()
        
        default_settings = Settings(user_id=new_user.id, recipient_email=email)
        db.session.add(default_settings)
        db.session.commit()
        
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "known_faces"), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "recordings"), exist_ok=True)
        
        login_user(new_user)
        return redirect(url_for('index'))
    
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}register.html', settings=settings)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            return redirect(url_for(get_dashboard_target(user)))
        else:
            flash('Login failed. Check details.', 'danger')
                
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}login.html')    

@app.route('/logout')
@login_required
def logout():
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
    if is_admin_user(current_user):
        return redirect(url_for('admin_dashboard'))
    
    user_cameras = Camera.query.filter_by(user_id=current_user.id).all()
    
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}index.html', cameras=user_cameras)    


@app.route('/admin')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    active_cameras = Camera.query.count()
    total_events = EventLog.query.count()
    total_recordings = 0
    preview_cameras = Camera.query.filter_by(is_public=True).order_by(Camera.id).limit(4).all()
    settings_entry = Settings.query.first()
    email_enabled = bool(settings_entry and settings_entry.email_alerts_enabled)

    since = datetime.utcnow() - timedelta(days=1)
    events_today = EventLog.query.filter(EventLog.timestamp >= since).count()
    new_users_today = User.query.filter(User.created_at >= since).count()

    recent_activity = []
    now = datetime.utcnow()
    for event in EventLog.query.order_by(EventLog.timestamp.desc()).limit(5).all():
        user_name = event.username or 'System'
        action = event.description or event.source_name or event.event_type or 'Event recorded'
        delta = now - (event.timestamp or now)
        if delta.days >= 1:
            time_label = f"{delta.days}d ago"
        elif delta.seconds >= 3600:
            time_label = f"{delta.seconds // 3600}h ago"
        elif delta.seconds >= 60:
            time_label = f"{delta.seconds // 60}m ago"
        else:
            time_label = 'just now'

        event_type = (event.event_type or '').lower()
        if 'critical' in event_type or 'alert' in event_type or 'unknown' in event_type or 'error' in event_type:
            dot_type = 'danger'
        elif 'login' in event_type or 'recognized' in event_type or 'person' in event_type:
            dot_type = 'success'
        elif 'logout' in event_type or 'settings' in event_type or 'recording' in event_type or 'camera' in event_type:
            dot_type = 'amber'
        else:
            dot_type = 'info'

        recent_activity.append({
            'user': user_name,
            'action': action,
            'time': time_label,
            'type': dot_type,
        })

    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(
        f'{template_path}admin_dashboard.html',
        total_users=total_users,
        active_cameras=active_cameras,
        total_events=total_events,
        total_recordings=total_recordings,
        storage_used='0 MB',
        storage_total='0 MB',
        events_today=events_today,
        new_users_today=new_users_today,
        uptime='Online',
        preview_cameras=preview_cameras,
        email_enabled=email_enabled,
        recent_activity=recent_activity,
    )

@app.route('/admin/request')
@admin_required
def admin_request():
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}admin_request.html')

@app.route('/admin/cctv')
@admin_required
def admin_cctv():
    camera_items = []
    for camera in Camera.query.filter_by(user_id=current_user.id).order_by(Camera.id).all():
        camera_items.append({
            'id': camera.id,
            'name': camera.name,
            'location': '',
            'status': 'online' if camera.is_active else 'offline',
            'motion_detected': False,
            'is_recording': False,
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}admin_cctv.html', cameras=camera_items)


@app.route('/admin/users')
@admin_required
def admin_users():
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}admin_users.html', users=User.query.order_by(User.id).all(), message=None)

@app.route('/admin/users/create', methods=['POST'])
@admin_required
def admin_create_user():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()

    if not username or not email or not password:
        flash('Username, email, and password are required.', 'danger')
        return redirect(url_for('admin_users'))

    if User.query.filter((User.email == email) | (User.username == username)).first():
        flash('A user with that email or username already exists.', 'danger')
        return redirect(url_for('admin_users'))

    new_user = User(username=username, email=email, password=generate_password_hash(password), role='user')
    db.session.add(new_user)
    db.session.commit()

    db.session.add(Settings(user_id=new_user.id, recipient_email=email))
    db.session.commit()

    os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'known_faces'), exist_ok=True)
    os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'recordings'), exist_ok=True)
    flash(f'User "{username}" created successfully.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle_role', methods=['POST'])
@admin_required
def admin_toggle_role(user_id):
    if user_id == current_user.id:
        flash('You cannot change your own role.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    user.role = 'admin' if get_user_role(user) != 'admin' else 'user'
    db.session.commit()
    flash(f'Role updated for {user.username}.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle_ban', methods=['POST'])
@admin_required
def admin_toggle_ban(user_id):
    if user_id == current_user.id:
        flash('You cannot ban yourself.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    user.role = 'banned' if get_user_role(user) != 'banned' else 'user'
    db.session.commit()
    flash(f'Access state updated for {user.username}.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    if user_id == current_user.id:
        flash('You cannot delete your own account.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    try:
        Camera.query.filter_by(user_id=user.id).delete()
        Settings.query.filter_by(user_id=user.id).delete()
        EventLog.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()
        flash(f'User {user.username} deleted.', 'success')
    except Exception as exc:
        db.session.rollback()
        flash(f'Unable to delete user: {exc}', 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/logs')
@admin_required
def admin_logs():
    logs = EventLog.query.order_by(EventLog.timestamp.desc()).all()
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}admin_logs.html', logs=logs, total_logs=len(logs), page=1)


@app.route('/admin/clear_events', methods=['POST'])
@admin_required
def admin_clear_events():
    EventLog.query.delete()
    db.session.commit()
    flash('All event logs were cleared.', 'success')
    return redirect(url_for('admin_dashboard'))


@app.route('/admin/system')
@admin_required
def admin_system():
    settings = Settings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = Settings(user_id=current_user.id)
        db.session.add(settings)
        db.session.commit()

    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(
        f'{template_path}admin_system.html',
        settings=settings,
        storage_pct=0,
        storage_used='0 MB',
        storage_free='0 MB',
        storage_total='0 MB',
        sys_info={'python_version': sys.version.split()[0], 'flask_version': 'unknown', 'cv2_version': cv2.__version__, 'uptime': 'Online', 'host': request.host},
        message=None,
    )


@app.route('/admin/system/save', methods=['POST'])
@admin_required
def admin_system_save():
    settings = Settings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = Settings(user_id=current_user.id)
        db.session.add(settings)

    settings.max_storage_gb = int(request.form.get('max_storage_gb', 50))
    settings.retention_days = int(request.form.get('retention_days', 30))
    settings.recording_path = request.form.get('recording_path', './recordings')

    settings.allow_registration = 'allow_registration' in request.form
    settings.require_disclaimer = 'require_disclaimer' in request.form
    settings.session_timeout_enabled = 'session_timeout_enabled' in request.form
    settings.session_timeout_minutes = int(request.form.get('session_timeout_minutes', 60))

    settings.yolo_enabled = 'yolo_enabled' in request.form
    settings.face_recognition_enabled = 'face_recognition_enabled' in request.form
    settings.object_detection_confidence = float(request.form.get('object_detection_confidence', 0.5))
    settings.face_recognition_confidence = float(request.form.get('face_recognition_confidence', 0.6))
    settings.active_classes = request.form.get('active_classes', 'fire,smoke')    
    settings.frame_process_interval = int(request.form.get('frame_process_interval', 3))
    

    db.session.commit()

    with stream_lock:
        if current_user.id in active_user_streams:
            for mgr in active_user_streams[current_user.id].values():
                mgr.update_settings(settings)

    flash('System settings updated.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_recordings', methods=['POST'])
@admin_required
def admin_clear_recordings():
    for user_dir in glob.glob(os.path.join(BASE_RECORDINGS_DIR, '*', 'recordings')):
        for filename in os.listdir(user_dir):
            if filename.endswith('.webm'):
                os.remove(os.path.join(user_dir, filename))
    flash('All recordings were cleared.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_events', methods=['POST'])
@admin_required
def admin_system_clear_events():
    EventLog.query.delete()
    db.session.commit()
    flash('All event logs were cleared.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_faces', methods=['POST'])
@admin_required
def admin_clear_faces():
    for user_dir in glob.glob(os.path.join(BASE_RECORDINGS_DIR, '*', 'known_faces')):
        shutil.rmtree(user_dir, ignore_errors=True)
        os.makedirs(user_dir, exist_ok=True)
    flash('All known faces were cleared.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/factory_reset', methods=['POST'])
@admin_required
def admin_factory_reset():
    EventLog.query.delete()
    Camera.query.delete()
    Settings.query.delete()
    db.session.commit()
    flash('Factory reset completed.', 'success')
    return redirect(url_for('admin_system'))

@app.route('/add_camera', methods=['POST'])
@login_required
def add_camera():
    source = request.form.get('source')
    name = request.form.get('name')
    redirect_target = url_for('admin_cctv') if is_admin_user(current_user) else url_for('index')

    if not source:
        flash('Camera source URL is required.', 'danger')
        return redirect(redirect_target)

    if source.isdigit():
        flash('Local Device IDs (0, 1, etc.) are not supported in Cloud Mode. Please provide an RTSP/HTTP URL.', 'danger')
        return redirect(redirect_target)

    new_cam = Camera(user_id=current_user.id, source=source, name=name, is_public=is_admin_user(current_user))
    db.session.add(new_cam)
    db.session.commit()
    flash('Camera added.', 'success')

    return redirect(redirect_target)

@app.route('/delete_camera/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    is_admin = is_admin_user(current_user)
    
    if camera.user_id != current_user.id and not is_admin:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('index'))
    
    if is_admin:
        redirect_target = url_for('admin_cctv')
    else:
        redirect_target = url_for('index')

    with stream_lock:
        user_key = camera.user_id if is_admin else current_user.id
        if user_key in active_user_streams:
            if camera.id in active_user_streams[user_key]:
                manager = active_user_streams[user_key][camera.id]
                manager.release()
                del active_user_streams[user_key][camera.id]
    
    try:
        db.session.delete(camera)
        db.session.commit()
        flash(f'Camera "{camera.name}" removed.', 'success')
    except Exception as e:
        flash(f'Error deleting camera: {e}', 'danger')
        
    return redirect(redirect_target)

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.user_id != current_user.id and not (is_admin_user(current_user) and camera.is_public):
        return "Unauthorized", 403

    stream_user_id = camera.user_id if is_admin_user(current_user) else current_user.id
    return Response(gen_frames(stream_user_id, camera), mimetype='multipart/x-mixed-replace; boundary=frame')

def gen_frames(user_id, camera):
    """Highly optimized frame generator. Reads pre-processed JPEGs directly from memory."""
    global active_user_streams
    
    manager = None
    with stream_lock:
        if user_id not in active_user_streams:
            active_user_streams[user_id] = {}
            
        if camera.id not in active_user_streams[user_id]:
            with app.app_context():
                user_settings = Settings.query.filter_by(user_id=user_id).first()
            manager = VideoStreamManager(user_id, camera.id, camera.source, user_settings)
            active_user_streams[user_id][camera.id] = manager
        else:
            manager = active_user_streams[user_id][camera.id]
            
    while True:
        # Instantly fetch pre-encoded JPEG bytes from the background processing thread
        frame_bytes = manager.get_latest_frame_bytes()
        if frame_bytes is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            # Rest to match typical camera feed rates and yield execution to other Flask request workers
            time.sleep(0.033)  # ~30 FPS limit
        else:
            time.sleep(0.01)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_settings = Settings.query.filter_by(user_id=current_user.id).first()
    # Ensure settings entry always exists for current user
    if not user_settings:
        user_settings = Settings(user_id=current_user.id)
        db.session.add(user_settings)
        db.session.commit()
    
    user_faces_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces")
    known_faces_list = []
    if os.path.exists(user_faces_dir):
        known_faces_list = [name for name in os.listdir(user_faces_dir) if os.path.isdir(os.path.join(user_faces_dir, name))]

    if request.method == 'POST':
        # Process settings attributes sent by settings.html
        user_settings.yolo_enabled = 'yolo_enabled' in request.form
        user_settings.face_recognition_enabled = 'face_rec_enabled' in request.form
        
        try:
            user_settings.object_detection_confidence = float(request.form.get('object_detection_confidence', 0.5))
        except (ValueError, TypeError):
            user_settings.object_detection_confidence = 0.5
            
        try:
            user_settings.face_recognition_confidence = float(request.form.get('face_recognition_confidence', 0.6))
        except (ValueError, TypeError):
            user_settings.face_recognition_confidence = 0.6

        # Correctly save the Frame Processing Interval on the user settings page as well
        try:
            user_settings.frame_process_interval = int(request.form.get('frame_process_interval', 3))
        except (ValueError, TypeError):
            user_settings.frame_process_interval = 3

        db.session.commit()
        flash("Settings Updated Successfully", "success")
        
        with stream_lock:
            if current_user.id in active_user_streams:
                for mgr in active_user_streams[current_user.id].values():
                    mgr.update_settings(user_settings)
        
        return redirect(url_for('settings'))
        
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}settings.html', settings=user_settings, known_faces=known_faces_list)

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
        safe_name = secure_filename(name)
        
        save_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces", safe_name)
        os.makedirs(save_dir, exist_ok=True)
        
        file.save(os.path.join(save_dir, filename))
        
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
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}recordings.html')

@app.route('/events')
@login_required
def events_page():
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}events.html')

@app.route('/api/recordings', methods=['GET'])
@login_required
def api_recordings():
    user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "recordings")
    recordings = []
    
    if os.path.exists(user_rec_dir):
        for filename in sorted(os.listdir(user_rec_dir), reverse=True):
            if filename.endswith('.webm'):
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
def api_clear_all_events():
    try:
        EventLog.query.filter_by(user_id=current_user.id).delete()
        db.session.commit()
        return jsonify({"success": True, "message": "All events cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/request_recording', methods=['POST'])
@login_required
def request_recording():
    # Handle manual recording logic here
    flash("Recording request processed", "success")
    return redirect(url_for('settings'))

@app.route('/cctv')
@login_required
def cctv():
    camera_items = []
    for camera in Camera.query.filter_by(user_id=current_user.id).order_by(Camera.id).all():
        camera_items.append({
            'id': camera.id,
            'name': camera.name,
            'location': '',
            'status': 'online' if camera.is_active else 'offline',
            'motion_detected': False,
            'is_recording': False,
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    is_mobile_device = is_mobile(request)
    template_path = 'mobile/' if is_mobile_device else ''
    return render_template(f'{template_path}cctv.html', cameras=camera_items)


# --- INITIALIZATION ---
def init_app():
    global yolo_model, yolo_model_object, detection_pool, ALL_YOLO_CLASS_NAMES, ACCELERATOR_DEVICE

    with app.app_context():
        ensure_database_schema()

    # Start database logger daemon thread
    log_worker = threading.Thread(target=db_logger_worker, daemon=True)
    log_worker.start()

    # ══════════════════════════════════════════════════════════════
    # DYNAMIC HARDWARE ACCELERATION ENGINE
    # ══════════════════════════════════════════════════════════════
    # We dynamically map models to NVIDIA CUDA, Intel OpenVINO GPU, or CPU.
    
    model_main = 'best.pt'
    model_secondary = 'yolo26n.pt'
    
    if CUDA_AVAILABLE:
        ACCELERATOR_DEVICE = "cuda"
        print(f"\n[GPU DETECTED] Success! Found NVIDIA GPU: {CUDA_DEVICE_NAME}")
        print(f"[ACCELERATOR] Using backend: CUDA.")
        
        # Check if TensorRT compiled engines exist (highly optimized)
        if os.path.exists('best.engine') and os.path.exists('yolo26n.engine'):
            model_main = 'best.engine'
            model_secondary = 'yolo26n.engine'
            print("[ACCELERATOR] Found pre-compiled TensorRT Engines (.engine). Using them for extreme speed!")
        else:
            print("[ACCELERATOR] Using standard PyTorch weight weights (.pt) on CUDA.")
            
    else:
        # Fallback to Intel OpenVINO configuration
        print("\n[GPU NOT DETECTED] NVIDIA GPU (CUDA) not active or not installed.")
        try:
            import openvino as ov
            core = ov.Core()
            available_devices = core.available_devices
            print(f"[OPENVINO] Available hardware on your system: {available_devices}")
            
            if "GPU" in available_devices:
                ACCELERATOR_DEVICE = "GPU"
                model_main = 'best_openvino_model'
                model_secondary = 'yolo26n_int8_openvino_model'
                print("[OPENVINO] Intel Integrated Graphics found. Loading OpenVINO models targeting Intel GPU.")
            else:
                ACCELERATOR_DEVICE = "CPU"
                model_main = 'best_openvino_model'
                model_secondary = 'yolo26n_int8_openvino_model'
                print("[OPENVINO] Defaulting to AVX-512 accelerated OpenVINO CPU execution.")
                
        except (ImportError, Exception) as e:
            ACCELERATOR_DEVICE = "cpu"
            print(f"[FALLBACK] No CUDA or OpenVINO found. Defaulting to CPU using PyTorch. Details: {e}")

    # Load computed configurations dynamically
    print(f"[LOADER] Loading Primary Model: '{model_main}' onto device: {ACCELERATOR_DEVICE}")
    yolo_model = YOLO(model_main)
    if yolo_model.names:
        ALL_YOLO_CLASS_NAMES = list(yolo_model.names.values())

    print(f"[LOADER] Loading Secondary Model: '{model_secondary}' onto device: {ACCELERATOR_DEVICE}")
    yolo_model_object = YOLO(model_secondary)

    # Core i5-1135G7 or standard processors have 4-8 physical cores.
    # Spawning workers safely avoiding thread subscription on CPU-bound HOG face tracking.
    optimal_workers = max(1, min(3, cpu_count() // 2))
    detection_pool = Pool(processes=optimal_workers)
    print(f"App Initialized with active structures. Multiprocessing Pool using {optimal_workers} worker threads.\n")


def main():
    parser = argparse.ArgumentParser(description='Run the smart security app')
    parser.add_argument('--create-admin', action='store_true', help='Create an admin account from the command line')
    parser.add_argument('--username', help='Username for the admin account')
    parser.add_argument('--email', help='Email address for the admin account')
    parser.add_argument('--password', help='Password for the admin account')
    args = parser.parse_args()

    if args.create_admin:
        try:
            admin_user = create_admin_user(args.username, args.email, args.password)
            print(f"Admin created successfully: {admin_user.email}")
            return 0
        except Exception as exc:
            print(f"Admin creation failed: {exc}")
            return 1

    init_app()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
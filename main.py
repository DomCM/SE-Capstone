import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import cv2
import time
import numpy as np
import sys
import glob
import shutil
import base64
import hashlib
import secrets
import gc
import torch
from urllib.parse import urlsplit, urlunsplit
from multiprocessing import Pool, cpu_count
from ultralytics import YOLO
import face_recognition
import logging
import click
from dotenv import load_dotenv

# Enforce PyTorch CPU single-threading to prevent thread memory proliferation
torch.set_num_threads(1)

# Force RTSP TCP transport, disable buffering, and enable low delay for FFmpeg capture
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"

logging.getLogger('opencv-python').setLevel(logging.ERROR)
os.environ['FFREPORT'] = 'file=/dev/null'
load_dotenv()
cv2.setNumThreads(1)

from flask import Flask, Response, render_template, request, jsonify, redirect, url_for, send_file, send_from_directory, flash, session
from models import db, User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest
import security
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
from sqlalchemy import inspect, text, or_
from cryptography.fernet import Fernet, InvalidToken
import pyotp
import qrcode
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_security.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SMTP_SERVER'] = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
app.config['SMTP_PORT'] = int(os.environ.get('SMTP_PORT', '587'))
app.config['SMTP_USERNAME'] = os.environ.get('SMTP_USERNAME', '')
app.config['SMTP_PASSWORD'] = os.environ.get('SMTP_PASSWORD', '')

db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

MODEL_PATH = 'models/best.pt' # main
MODEL_PATH_OBJECT = 'models/yolo26n.pt'  # secondary
BASE_RECORDINGS_DIR = "users_data"
OVERLAP_PIXELS = 44
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'webm'}

security.configure(app, db, (User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest), BASE_RECORDINGS_DIR)

detection_pool = None
yolo_model = None
yolo_model_object = None
ALL_YOLO_CLASS_NAMES = []

active_user_streams = {}
active_physical_streams = {}
stream_lock = threading.Lock()
STREAM_HEALTH_TIMEOUT_SECONDS = 3

face_cache = {}
MAX_DETECTION_POOL_WORKERS = max(1, min(2, cpu_count() or 1))
PROCESSING_INTERVAL_SECONDS = 0.1
detection_pool = None


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


def normalize_camera_source(source):
    """Return a stable key for equivalent network camera URLs."""
    source = str(source or '').strip()
    if not source:
        return source

    parsed = urlsplit(source)
    if not parsed.scheme or not parsed.netloc:
        return source

    hostname = (parsed.hostname or '').lower()
    try:
        port = parsed.port
    except ValueError:
        port = None
    default_port = (parsed.scheme.lower() == 'rtsp' and port == 554) or (
        parsed.scheme.lower() == 'http' and port == 80
    ) or (parsed.scheme.lower() == 'https' and port == 443)
    netloc = hostname
    if parsed.username:
        netloc = parsed.username + (':' + parsed.password if parsed.password else '') + '@' + netloc
    if port and not default_port:
        netloc += f':{port}'

    path = parsed.path.rstrip('/') or '/'
    return urlunsplit((parsed.scheme.lower(), netloc, path, parsed.query, ''))


def stream_scope_key(user_id, camera):
    """Share public sources globally; isolate private sources by owner."""
    source_key = normalize_camera_source(camera.source)
    return source_key if camera.is_public else (user_id, source_key)

#main

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

ADMIN_EMAILS = [email.strip().lower() for email in os.environ.get('ADMIN_EMAILS', '').split(',') if email.strip()]


from security import (
    get_user_role, is_admin_user, get_client_ip, record_audit_event,
    get_dashboard_target, encrypt_totp_secret, decrypt_totp_secret,
    get_or_create_totp_secret, send_security_email, create_email_otp,
    clear_audit_events_for_user, ensure_database_schema, create_admin_user,
    AUDIT_EVENT_TYPES,
)
from reports import build_report


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

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
                        del image
                    except Exception as e:
                        print(f"Error loading face {filename}: {e}")
    
    face_cache[user_id] = (known_encodings, known_names)
    return known_encodings, known_names

def detect_faces_in_chunk(cropped_image, bbox, scale_factor, upsample_amount, known_encodings, known_names, face_confidence=0.6):
    """
    Optimized worker function. Processes facial recognition on a pre-cropped ROI.
    """
    if cropped_image is None or cropped_image.size == 0 or bbox is None:
        return []

    x1, y1, x2, y2 = map(int, bbox)

    if scale_factor != 1.0:
        cropped_image = cv2.resize(cropped_image, (0, 0), fx=scale_factor, fy=scale_factor)

    rgb_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    chunk_face_locations = face_recognition.face_locations(
        rgb_cropped_image,
        model="hog",
        number_of_times_to_upsample=upsample_amount,
    )
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
    
    del rgb_cropped_image
    return results


def run_roi_detection(roi_tasks):
    """Run face-detection ROI tasks using ThreadPoolExecutor to prevent memory pickling overhead."""
    if not roi_tasks:
        return []

    if len(roi_tasks) == 1 or detection_pool is None:
        return [detect_faces_in_chunk(*task) for task in roi_tasks]

    try:
        futures = [detection_pool.submit(detect_faces_in_chunk, *task) for task in roi_tasks]
        return [f.result() for f in futures]
    except Exception:
        return [detect_faces_in_chunk(*task) for task in roi_tasks]


def send_email_alert(user_settings, subject, body, image_frame=None):
    if not user_settings or not user_settings.email_alerts_enabled:
        return

    sender = app.config['SMTP_USERNAME']
    password = app.config['SMTP_PASSWORD']
    if not sender or not password or not user_settings.recipient_email:
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = user_settings.recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        if image_frame is not None:
            success, encoded_image = cv2.imencode('.jpg', image_frame)
            if success:
                msg.attach(MIMEImage(encoded_image.tobytes(), name="alert.jpg"))

        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        print(f"Email sent to {user_settings.recipient_email}")
    except Exception as e:
        print(f"Email failed: {e}")

#video
class VideoStreamManager:
    def __init__(self, user_id, camera_id, source, settings, camera_name=None):
        self.user_id = user_id
        self.camera_id = camera_id
        self.camera_name = camera_name or f'Cam {camera_id}'
        self.video_source = source
        self.cap = None
        self.last_connection_attempt = 0
        self.connection_retry_delay = 5
        
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.processing_lock = threading.Lock()
        self.reader_thread = None
        self.reader_thread_stop = False
        self.stream_stop_event = threading.Event()
        self.last_frame_at = 0.0
        self.processing_thread = None
        
        self.processed_frame = None
        self.frame_id = 0
        self.processed_frame_lock = threading.Lock()
        self.subscriber_count = 0
        
        self.settings_cache = self._cache_settings(settings)
        
        self.frame_count = 0
        self.fire_alert_active = False
        self.face_detections = []
        self.yolo_detections = []
        self.recording_writer = None
        self.is_recording = False
        
        # Multitracking collection structure
        self.trackers = []          
        self.tracker_active = False
        self.tracker_lost = False
        self.tracker_frame_count = 0
        
        self.known_encodings, self.known_names = get_user_face_data(user_id)
        self.debug_tracker = False
        self._recent_event_cache = {}
        self._last_full_scan_time = 0.0

    def start(self):
        """Start processing so the worker can establish the source connection."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()

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
                'recipient_email': settings_obj.recipient_email,
                'scale_down_amount': settings_obj.scale_down_amount,
                'frame_process_interval': settings_obj.frame_process_interval,
            }
        return {}

    def update_settings(self, settings_obj):
        """Update the settings cache with new values."""
        self.settings_cache = self._cache_settings(settings_obj)

    def _reset_tracker(self):
        if self.debug_tracker and self.tracker_active:
            print(f"[TRACKERS] Resetting/Clearing trackers at frame {self.frame_count}")
        self.trackers = []
        self.tracker_active = False
        self.tracker_lost = True

    def _initialize_trackers(self, frame, bboxes):
        """Initialize independent trackers for multiple detected humans."""
        self.trackers = []
        if frame is None or not bboxes:
            self.tracker_active = False
            self.tracker_lost = True
            return

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            width = max(1, int(x2 - x1))
            height = max(1, int(y2 - y1))
            try:
                try:
                    tracker = cv2.legacy.TrackerKCF.create()
                    tracker_name = "KCF (legacy)"
                except (AttributeError, NameError):
                    tracker = cv2.TrackerKCF.create()
                    tracker_name = "KCF"
                    
                ok = tracker.init(frame, (int(x1), int(y1), width, height))
                if ok:
                    self.trackers.append({
                        "tracker": tracker,
                        "bbox": (int(x1), int(y1), int(x2), int(y2))
                    })
            except Exception as e:
                print(f"Tracker init error: {e}")

        if self.trackers:
            self.tracker_active = True
            self.tracker_lost = False
            self.tracker_frame_count = 0
        else:
            self.tracker_active = False
            self.tracker_lost = True

    def _update_trackers(self, frame):
        """Update all active trackers and prune any that failed/lost track."""
        if not self.trackers or frame is None:
            self.tracker_active = False
            self.tracker_lost = True
            return []

        updated_bboxes = []
        active_trackers = []

        for item in self.trackers:
            tracker = item["tracker"]
            try:
                ok, bbox = tracker.update(frame)
                if ok:
                    x, y, w, h = [int(v) for v in bbox]
                    new_bbox = (x, y, x + w, y + h)
                    item["bbox"] = new_bbox
                    active_trackers.append(item)
                    updated_bboxes.append(new_bbox)
            except Exception as e:
                print(f"Tracker update error: {e}")

        self.trackers = active_trackers
        if self.trackers:
            self.tracker_active = True
            self.tracker_lost = False
            self.tracker_frame_count += 1
        else:
            self.tracker_active = False
            self.tracker_lost = True

        return updated_bboxes

    def _open_stream(self):
        """Lazily open the video stream with error handling, TCP forcing, and keyframe flushing."""
        try:
            with self.frame_lock:
                self.current_frame = None
            self._reset_tracker()

            # Ensure FFmpeg options are enforced for RTSP/HTTP network streams
            if isinstance(self.video_source, str) and self.video_source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
                self.cap = cv2.VideoCapture(self.video_source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(self.video_source)
            
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Set open and read timeouts if supported by OpenCV backend
                if hasattr(cv2, 'CAP_PROP_OPEN_TIMEOUT_MSEC'):
                    self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                if hasattr(cv2, 'CAP_PROP_READ_TIMEOUT_MSEC'):
                    self.cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)

                for _ in range(12):
                    self.cap.grab()

                self.reader_thread_stop = False
                self.reader_thread = threading.Thread(target=self._read_frames_worker, daemon=True)
                self.reader_thread.start()
                if self.processing_thread is None or not self.processing_thread.is_alive():
                    self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
                    self.processing_thread.start()
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
            if ret and frame is not None and frame.size > 0:
                with self.frame_lock:
                    self.current_frame = frame
                self.last_frame_at = time.monotonic()
            else:
                with self.frame_lock:
                    self.current_frame = None
                self.last_frame_at = 0.0
                self.reader_thread_stop = True
                if self.cap:
                    self.cap.release()
                self.cap = None
                self._reset_tracker()
                break

    def is_connected(self):
        """Return true only while the source has delivered a recent frame."""
        cap = self.cap
        return (
            cap is not None
            and cap.isOpened()
            and self.last_frame_at > 0
            and time.monotonic() - self.last_frame_at <= STREAM_HEALTH_TIMEOUT_SECONDS
        )

    def _processing_worker(self):
        """Process the source independently of whether a page is currently open."""
        loop_counter = 0
        while not self.stream_stop_event.is_set():
            processed_frame = self._process_current_frame()
            if processed_frame is not None:
                with self.processed_frame_lock:
                    self.processed_frame = processed_frame
                    self.frame_id += 1

            loop_counter += 1
            if loop_counter >= 100:
                gc.collect()
                loop_counter = 0

            if not self.stream_stop_event.is_set():
                self.stream_stop_event.wait(PROCESSING_INTERVAL_SECONDS)

    def get_processed_frame(self, last_seen_id):
        """Only returns a frame copy if a new frame version is available."""
        with self.processed_frame_lock:
            if self.processed_frame is None or self.frame_id <= last_seen_id:
                return self.frame_id, None
            return self.frame_id, self.processed_frame.copy()

    def process_frame(self):
        with self.processed_frame_lock:
            if self.processed_frame is None:
                return None
            return self.processed_frame.copy()

    def _process_current_frame(self):
        if self.stream_stop_event.is_set():
            return None

        with self.processing_lock:
            if self.stream_stop_event.is_set():
                return None

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

            if not self.is_connected():
                return None

            with self.frame_lock:
                frame = self.current_frame

            if frame is None:
                return None

            annotated_frame = frame.copy()
        
        scale_down = self.settings_cache.get('scale_down_amount', 2)
        process_interval = max(1, self.settings_cache.get('frame_process_interval', 3))
        active_classes = self.settings_cache.get('active_classes', 'fire,smoke').split(',')
        
        global face_cache
        if self.user_id not in face_cache:
             self.known_encodings, self.known_names = get_user_face_data(self.user_id)

        # 1. KCF TRACKING FOR INTERMEDIATE FRAMES
        if self.tracker_active:
            tracked_boxes = self._update_trackers(frame)
            
            if tracked_boxes:
                for x1, y1, x2, y2 in tracked_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                for (top, right, bottom, left), name in self.face_detections:
                    color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(annotated_frame, f"Verified: {name}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(annotated_frame, f"Tracking Active ({len(tracked_boxes)} Persons)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                if self.tracker_frame_count >= process_interval:
                    self._reset_tracker()

                self.frame_count += 1
                return annotated_frame
            else:
                self._reset_tracker()

        # 2. FRAME SKIPPING FOR DETECTIONS
        if self.frame_count % process_interval != 0:
            self.frame_count += 1
            return annotated_frame

        # 3. FULL COMPUTER VISION SCAN PASS
        critical_detected = False
        detected_crit_names = []
        self.yolo_detections = []
        human_boxes = []

        # Primary YOLO Model
        if self.settings_cache.get('yolo_enabled', True) and yolo_model:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            with torch.no_grad():
                results = yolo_model.predict(frame, imgsz=320, conf=obj_conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0].item())
                    name = r.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                    
                    is_crit = name in active_classes
                    if is_crit:
                        critical_detected = True
                        detected_crit_names.append((name, confidence))
                    
                    self.yolo_detections.append((x1, y1, x2, y2, name, is_crit))
                    if is_human_detection_name(name):
                        human_boxes.append((x1, y1, x2, y2, confidence))

        # Secondary YOLO Model
        if self.settings_cache.get('yolo_object_enabled', True) and yolo_model_object:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            with torch.no_grad():
                results = yolo_model_object.predict(frame, imgsz=320, conf=obj_conf, verbose=False)
            for r in results:
                for box in r.boxes:
                    name = r.names.get(int(box.cls[0].item()))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                    self.yolo_detections.append((x1, y1, x2, y2, name, False))
                    if is_human_detection_name(name):
                        human_boxes.append((x1, y1, x2, y2, confidence))

        # Draw YOLO detections
        for (x1, y1, x2, y2, name, is_crit) in self.yolo_detections:
            color = (0, 0, 255) if is_crit else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Parallel Face Recognition on Human ROIs
        if human_boxes and self.settings_cache.get('face_recognition_enabled', True):
            scale_factor = 1.0 / scale_down
            face_conf = self.settings_cache.get('face_recognition_confidence', 0.6)
            
            roi_tasks = []
            frame_h, frame_w = frame.shape[:2]

            for bbox in human_boxes:
                x1, y1, x2, y2 = map(int, bbox[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w, x2), min(frame_h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

                cropped_roi = frame[y1:y2, x1:x2]
                roi_tasks.append((cropped_roi, (x1, y1, x2, y2), scale_factor, 1, self.known_encodings, self.known_names, face_conf))

            try:
                all_results = run_roi_detection(roi_tasks)

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

            # Draw startup face bounds
            for (top, right, bottom, left), name in self.face_detections:
                color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(annotated_frame, name, (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            bboxes_to_track = [box[:4] for box in human_boxes]
            self._initialize_trackers(frame, bboxes_to_track)

        # Handle Alerts
        if critical_detected and not self.fire_alert_active:
            detected_names = list(dict.fromkeys(name for name, _ in detected_crit_names))
            highest_confidence = max((confidence for _, confidence in detected_crit_names), default=None)
            event_type = f"CRITICAL ALERT: {', '.join(detected_names)}"
            msg = f"Detected: {', '.join(detected_names)}"
            self.log_db_event(event_type, msg, highest_confidence)
            class SettingsObj:
                pass
            settings_obj = SettingsObj()
            for k, v in self.settings_cache.items():
                setattr(settings_obj, k, v)
            threading.Thread(target=send_email_alert, args=(settings_obj, "CRITICAL ALERT", msg, annotated_frame), daemon=True).start()
        
        self.fire_alert_active = critical_detected

        # Record stream frames
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

    def log_db_event(self, event_type, desc, confidence=None):
        dedupe_key = (event_type, (desc or '')[:180])
        now = time.monotonic()
        last_logged = self._recent_event_cache.get(dedupe_key)
        if last_logged and now - last_logged < 15:
            return
        self._recent_event_cache[dedupe_key] = now

        with app.app_context():
            try:
                log = EventLog(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    source_name=self.camera_name,
                    event_type=event_type,
                    description=desc,
                    confidence=confidence,
                )
                db.session.add(log)
                db.session.commit()
            finally:
                db.session.remove()

    def release(self):
        self.stream_stop_event.set()
        self.stop_recording()
        self.reader_thread_stop = True
        if self.reader_thread:
            self.reader_thread.join(timeout=2)
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        if self.cap:
            self.cap.release()

#routes

def is_mobile(request):
    """
    Checks the User-Agent header to determine if the request is coming from a mobile device.
    """
    if 'User-Agent' not in request.headers:
        return False
    
    user_agent = request.headers['User-Agent'].lower()
    mobile_keywords = ['android', 'iphone', 'ipad', 'ipod', 'blackberry', 'windows phone', 'opera mini']
    for keyword in mobile_keywords:
        if keyword in user_agent:
            return True
    return False

#gatekeep start
@app.before_request
def check_privacy_agreement():
    allowed_endpoints = ['disclaimer', 'accept_terms', 'static']
    if request.endpoint in allowed_endpoints:
        return

    if not session.get('privacy_agreed'):
        return redirect(url_for('disclaimer'))

    enrollment_endpoints = {'security_setup', 'logout', 'static', 'disclaimer', 'accept_terms'}
    if (current_user.is_authenticated and current_user.totp_secret
            and not current_user.totp_enabled
            and request.endpoint not in enrollment_endpoints):
        return redirect(url_for('security_setup'))


@app.after_request
def prevent_authenticated_page_caching(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/accept_terms', methods=['POST'])
def accept_terms():
    session['privacy_agreed'] = True
    return redirect(url_for('login'))
#gatekeep end

def admin_required(view_func):
    @wraps(view_func)
    @login_required
    def wrapped(*args, **kwargs):
        if not is_admin_user(current_user):
            flash('You do not have permission to access the admin area.', 'danger')
            return redirect(url_for(get_dashboard_target(current_user)))
        return view_func(*args, **kwargs)
    return wrapped


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        if user:
            code = create_email_otp(user)
            send_security_email(
                user,
                'Home Detection Security password reset code',
                f'Your password reset code is {code}. It expires in 10 minutes.',
            )
        session['recovery_email'] = email
        return redirect(url_for('verify_otp', purpose='password_reset'))
    return render_template('forgot_password.html')


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    purpose = request.args.get('purpose') or session.get('otp_purpose')
    if purpose not in {'login', 'password_reset'}:
        return redirect(url_for('login'))
    session['otp_purpose'] = purpose

    if request.method == 'POST':
        code = request.form.get('otp', '').strip()
        if not code.isdigit() or len(code) != 6:
            flash('Enter the 6-digit verification code.', 'danger')
            return render_template('verify_otp.html', purpose=purpose, email=session.get('recovery_email'))

        if purpose == 'login':
            user = db.session.get(User, session.get('pending_login_user_id'))
            secret = decrypt_totp_secret(user.totp_secret) if user else None
            valid = bool(user and user.totp_enabled and secret and pyotp.TOTP(secret).verify(code))
        else:
            email = session.get('recovery_email')
            user = User.query.filter_by(email=email).first() if email else None
            challenge = OtpChallenge.query.filter_by(
                user_id=user.id if user else 0, purpose='password_reset', used_at=None
            ).order_by(OtpChallenge.created_at.desc()).first() if user else None
            valid = bool(
                challenge and challenge.expires_at > datetime.utcnow()
                and challenge.attempts < 5
                and check_password_hash(challenge.code_hash, code)
            )
            if challenge:
                challenge.attempts += 1
                if valid:
                    challenge.used_at = datetime.utcnow()
                db.session.commit()

        if valid:
            if purpose == 'login':
                user.last_login = datetime.utcnow()
                db.session.commit()
                login_user(user)
                session.pop('pending_login_user_id', None)
                session.pop('otp_purpose', None)
                return redirect(url_for(get_dashboard_target(user)))
            session['password_reset_user_id'] = user.id
            session.pop('otp_purpose', None)
            return redirect(url_for('reset_password'))

        flash('That verification code is invalid or expired.', 'danger')

    return render_template('verify_otp.html', purpose=purpose, email=session.get('recovery_email'))


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    user = db.session.get(User, session.get('password_reset_user_id'))
    if not user:
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirmation = request.form.get('confirm_password', '')
        if len(password) < 12 or password != confirmation:
            flash('Use a strong password and make sure both fields match.', 'danger')
            return render_template('reset_password.html')
        user.password = generate_password_hash(password)
        db.session.commit()
        session.pop('password_reset_user_id', None)
        return redirect(url_for('password_success'))
    return render_template('reset_password.html')


@app.route('/password-success')
def password_success():
    return render_template('password_success.html')


@app.route('/security/setup', methods=['GET', 'POST'])
@login_required
def security_setup():
    secret = get_or_create_totp_secret(current_user)
    if request.method == 'POST':
        code = request.form.get('otp', '').strip()
        if pyotp.TOTP(secret).verify(code):
            current_user.totp_enabled = True
            db.session.commit()
            flash('Authenticator verification is now enabled.', 'success')
            return redirect(url_for(get_dashboard_target(current_user)))
        flash('Enter the current 6-digit code from your authenticator.', 'danger')

    uri = pyotp.TOTP(secret).provisioning_uri(
        name=current_user.email, issuer_name='Home Detection Security'
    )
    qr = qrcode.make(uri)
    image = BytesIO()
    qr.save(image, format='PNG')
    qr_data = base64.b64encode(image.getvalue()).decode()
    return render_template('security_setup.html', qr_data=qr_data, secret=secret)


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
        
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            role='user',
            totp_secret=encrypt_totp_secret(pyotp.random_base32()),
        )
        db.session.add(new_user)
        db.session.commit()

        default_settings = Settings(user_id=new_user.id, recipient_email=email)
        db.session.add(default_settings)
        db.session.commit()

        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "known_faces"), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "recordings"), exist_ok=True)

        record_audit_event(new_user.id, 'create', f"Created account for {new_user.username}", 'Auth', get_client_ip())

        login_user(new_user)
        return redirect(url_for('security_setup'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        logout_user()
        return redirect(url_for('login'))

    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password):
            if user.totp_secret and user.totp_enabled:
                session['pending_login_user_id'] = user.id
                session['otp_purpose'] = 'login'
                return redirect(url_for('verify_otp', purpose='login'))
            if user.totp_secret and not user.totp_enabled:
                login_user(user)
                return redirect(url_for('security_setup'))
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            record_audit_event(user.id, 'login', f"{user.username} logged in", 'Auth', get_client_ip())
            return redirect(url_for(get_dashboard_target(user)))
        else:
            flash('Login failed. Incorrect username or password.', 'danger')
                
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    user = current_user
    with stream_lock:
        user_streams = list(active_user_streams.get(current_user.id, {}).items())
    for camera_id, manager in user_streams:
        release_stream(current_user.id, camera_id, manager.scope_key, manager)

    record_audit_event(user.id, 'logout', f"{user.username} logged out", 'Auth', get_client_ip())
    logout_user()
    return redirect(url_for('login'))

@app.route('/')
@login_required
def index():
    if is_admin_user(current_user):
        return redirect(url_for('admin_dashboard'))
    
    user_cameras = Camera.query.filter(
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
    ).order_by(Camera.id).all()
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    week_start = datetime.utcnow() - timedelta(days=7)
    incidents_this_month = EventLog.query.filter(
        EventLog.user_id == current_user.id,
        EventLog.timestamp >= month_start,
    ).count()
    alerts_this_week = EventLog.query.filter(
        EventLog.user_id == current_user.id,
        EventLog.timestamp >= week_start,
        EventLog.event_type.ilike('%alert%'),
    ).count()
    high_severity_alerts = EventLog.query.filter(
        EventLog.user_id == current_user.id,
        EventLog.timestamp >= month_start,
        EventLog.event_type.ilike('%critical%'),
    ).count()
    recording_requests = RecordingRequest.query.filter_by(user_id=current_user.id).order_by(RecordingRequest.created_at.desc()).all()
    
    return render_template(
        'index.html',
        cameras=user_cameras,
        incidents_this_month=incidents_this_month,
        alerts_this_week=alerts_this_week,
        high_severity_alerts=high_severity_alerts,
        recording_requests=recording_requests,
    )


@app.route('/recording_requests', methods=['POST'])
@login_required
def create_recording_request():
    camera_id = request.form.get('camera_id', type=int)
    date_needed = request.form.get('date_needed', '').strip()
    time_range = request.form.get('time_range', '').strip()
    reason = request.form.get('reason', '').strip()
    camera = Camera.query.filter(
        Camera.id == camera_id,
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True)),
    ).first()

    if not camera or not date_needed or not time_range or not reason:
        flash('Camera, date, time range, and reason are required.', 'danger')
        return redirect(url_for('index'))

    try:
        datetime.strptime(date_needed, '%Y-%m-%d')
    except ValueError:
        flash('Please provide a valid request date.', 'danger')
        return redirect(url_for('index'))

    recording_request = RecordingRequest(
        user_id=current_user.id,
        camera_id=camera.id,
        date_needed=date_needed,
        time_range=time_range[:100],
        reason=reason[:1000],
    )
    db.session.add(recording_request)
    db.session.commit()
    flash('Recording request submitted.', 'success')
    return redirect(url_for('index'))


@app.route('/recording_requests/<int:request_id>/download')
@login_required
def download_recording_request(request_id):
    recording_request = RecordingRequest.query.filter_by(id=request_id, user_id=current_user.id).first_or_404()
    if recording_request.status != 'fulfilled' or not recording_request.video_path or not os.path.isfile(recording_request.video_path):
        return 'Recording is not available.', 404
    return send_file(recording_request.video_path, as_attachment=True, download_name=recording_request.video_filename)

@app.route('/cctv')
@login_required
def cctv():
    user_cameras = []
    cameras = Camera.query.filter(
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
    ).order_by(Camera.id).all()
    for camera in cameras:
        stream_owner_id = camera.user_id if camera.is_public else current_user.id
        user_cameras.append({
            'id': camera.id,
            'name': camera.name,
            'location': '',
            'status': 'online' if camera.is_active else 'offline',
            'motion_detected': False,
            'is_recording': bool(
                active_user_streams.get(stream_owner_id, {}).get(camera.id)
                and active_user_streams[stream_owner_id][camera.id].is_recording
            ),
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    
    return render_template('cctv.html', cameras=user_cameras)


@app.route('/admin')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_cameras = Camera.query.count()
    active_cameras = Camera.query.filter_by(is_active=True).count()
    total_events = EventLog.query.count()
    pending_requests_count = RecordingRequest.query.filter_by(status='pending').count()

    total_recordings = 0
    storage_used_bytes = 0
    for root, _, filenames in os.walk(BASE_RECORDINGS_DIR):
        if os.path.basename(root) != 'recordings':
            continue
        for filename in filenames:
            if os.path.splitext(filename)[1].lower().lstrip('.') in VIDEO_EXTENSIONS:
                total_recordings += 1
                storage_used_bytes += os.path.getsize(os.path.join(root, filename))

    if storage_used_bytes < 1024 * 1024:
        storage_used = f'{storage_used_bytes / 1024:.1f} KB'
    elif storage_used_bytes < 1024 * 1024 * 1024:
        storage_used = f'{storage_used_bytes / (1024 * 1024):.1f} MB'
    else:
        storage_used = f'{storage_used_bytes / (1024 * 1024 * 1024):.1f} GB'

    since = datetime.utcnow() - timedelta(days=1)
    events_today = EventLog.query.filter(EventLog.timestamp >= since).count()
    new_users_today = User.query.filter(User.created_at >= since).count()

    alert_events_today = EventLog.query.filter(
        EventLog.timestamp >= since,
        ~EventLog.event_type.in_(list(AUDIT_EVENT_TYPES)),
    ).all()
    zone_counts = {}
    hour_counts = {}
    for event in alert_events_today:
        zone = event.source_name or 'Unknown source'
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        if event.timestamp:
            hour_counts[event.timestamp.strftime('%I %p').lstrip('0')] = hour_counts.get(
                event.timestamp.strftime('%I %p').lstrip('0'), 0
            ) + 1

    total_zone_alerts = sum(zone_counts.values())
    alert_zones = [
        {
            'name': zone,
            'count': count,
            'ratio': round(count / total_zone_alerts * 100) if total_zone_alerts else 0,
        }
        for zone, count in sorted(zone_counts.items(), key=lambda item: item[1], reverse=True)
    ]
    peak_alert_time = max(hour_counts, key=hour_counts.get) if hour_counts else 'No alerts'

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
            'title': action,
            'detail': user_name,
            'time': time_label,
            'type': dot_type,
        })

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        active_cameras=active_cameras,
        total_cameras=total_cameras,
        total_events=total_events,
        total_recordings=total_recordings,
        storage_used=storage_used,
        events_today=events_today,
        new_users_today=new_users_today,
        camera_coverage=round(active_cameras / total_cameras * 100) if total_cameras else 0,
        alert_zones=alert_zones,
        alert_count_today=len(alert_events_today),
        highest_alert_zone=alert_zones[0]['name'] if alert_zones else 'No alerts',
        peak_alert_time=peak_alert_time,
        pending_requests_count=pending_requests_count,
        current_year=datetime.utcnow().year,
        recent_activity=recent_activity,
    )


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
            'is_public': camera.is_public,
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    return render_template('admin_cctv.html', cameras=camera_items)


@app.route('/admin/users')
@admin_required
def admin_users():
    users = User.query.order_by(User.id).all()
    return render_template('admin_users.html', users=users, message=None)


@app.route('/admin/users/create', methods=['POST'])
@admin_required
def admin_create_user():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip()
    password = request.form.get('password', '').strip()
    requested_role = request.form.get('role', 'user').strip().lower()

    if not username or not email or not password:
        flash('Username, email, and password are required.', 'danger')
        return redirect(url_for('admin_users'))

    if User.query.filter((User.email == email) | (User.username == username)).first():
        flash('A user with that email or username already exists.', 'danger')
        return redirect(url_for('admin_users'))

    if requested_role not in {'user', 'admin'}:
        requested_role = 'user'

    new_user = User(
        username=username,
        email=email,
        password=generate_password_hash(password),
        role=requested_role,
        totp_secret=encrypt_totp_secret(pyotp.random_base32()),
        totp_enabled=False,
    )
    db.session.add(new_user)
    db.session.commit()

    db.session.add(Settings(user_id=new_user.id, recipient_email=email))
    db.session.commit()

    os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'known_faces'), exist_ok=True)
    os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'recordings'), exist_ok=True)

    record_audit_event(current_user.id, 'create', f"Created user \"{username}\" ({requested_role})", 'Admin', get_client_ip())
    flash(f'User "{username}" created successfully.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle_role', methods=['POST'])
@admin_required
def admin_toggle_role(user_id):
    if user_id == current_user.id:
        flash('You cannot change your own role.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    new_role = 'admin' if get_user_role(user) != 'admin' else 'user'
    user.role = new_role
    db.session.commit()
    record_audit_event(current_user.id, 'settings', f"Updated role for {user.username} to {new_role}", 'Admin', get_client_ip())
    flash(f'Role updated for {user.username}.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/toggle_ban', methods=['POST'])
@admin_required
def admin_toggle_ban(user_id):
    if user_id == current_user.id:
        flash('You cannot ban yourself.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    new_role = 'banned' if get_user_role(user) != 'banned' else 'user'
    user.role = new_role
    db.session.commit()
    record_audit_event(current_user.id, 'settings', f"Updated access state for {user.username} to {new_role}", 'Admin', get_client_ip())
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
        record_audit_event(current_user.id, 'delete', f"Deleted user \"{user.username}\"", 'Admin', get_client_ip())
        flash(f'User {user.username} deleted.', 'success')
    except Exception as exc:
        db.session.rollback()
        flash(f'Unable to delete user: {exc}', 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/logs')
@admin_required
def admin_logs():
    query = EventLog.query.join(User, EventLog.user_id == User.id, isouter=True)

    user_filter = (request.args.get('user') or '').strip()
    action_filter = (request.args.get('action') or '').strip()
    from_date = (request.args.get('from') or '').strip()
    to_date = (request.args.get('to') or '').strip()

    if user_filter:
        search = f"%{user_filter}%"
        query = query.filter(or_(User.username.ilike(search), User.email.ilike(search)))

    if action_filter:
        query = query.filter(EventLog.event_type.ilike(action_filter))

    if from_date:
        try:
            from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            query = query.filter(EventLog.timestamp >= from_dt)
        except ValueError:
            pass

    if to_date:
        try:
            to_dt = datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(EventLog.timestamp < to_dt)
        except ValueError:
            pass

    query = query.filter(EventLog.event_type.in_(list(AUDIT_EVENT_TYPES)))
    logs = query.order_by(EventLog.timestamp.desc()).all()
    return render_template('admin_logs.html', logs=logs, total_logs=len(logs), page=1)


@app.route('/admin/reports')
@admin_required
def admin_reports():
    report_context = build_report(
        EventLog,
        report_type=request.args.get('type', 'village'),
        timeframe=request.args.get('timeframe', '7d'),
        zone=request.args.get('zone', 'all'),
        severity=request.args.get('severity', 'all'),
    )
    return render_template('admin_reports.html', **report_context)


@app.route('/admin/request')
@admin_required
def admin_request():
    requests = []
    for recording_request in RecordingRequest.query.order_by(RecordingRequest.created_at.desc()).all():
        requests.append({
            'id': recording_request.id,
            'username': recording_request.user.username,
            'camera_name': recording_request.camera.name,
            'date_needed': recording_request.date_needed,
            'time_range': recording_request.time_range,
            'reason': recording_request.reason,
            'created_at': recording_request.created_at.strftime('%Y-%m-%d %H:%M'),
            'status': recording_request.status,
            'rejection_reason': recording_request.rejection_reason,
            'video_filename': recording_request.video_filename,
        })
    stats = {status: RecordingRequest.query.filter_by(status=status).count()
             for status in ('pending', 'approved', 'fulfilled', 'rejected')}
    return render_template('admin_request.html', requests=requests, stats=stats)


@app.route('/admin/request/<int:request_id>/approve', methods=['POST'])
@admin_required
def approve_recording_request(request_id):
    recording_request = RecordingRequest.query.get_or_404(request_id)
    if recording_request.status != 'pending':
        flash('Only pending requests can be approved.', 'danger')
    else:
        recording_request.status = 'approved'
        db.session.commit()
        flash('Recording request approved.', 'success')
    return redirect(url_for('admin_request'))


@app.route('/admin/request/<int:request_id>/reject', methods=['POST'])
@admin_required
def reject_recording_request(request_id):
    recording_request = RecordingRequest.query.get_or_404(request_id)
    rejection_reason = request.form.get('rejection_reason', '').strip()
    if recording_request.status != 'pending':
        flash('Only pending requests can be rejected.', 'danger')
    elif not rejection_reason:
        flash('A rejection reason is required.', 'danger')
    else:
        recording_request.status = 'rejected'
        recording_request.rejection_reason = rejection_reason[:1000]
        db.session.commit()
        flash('Recording request rejected.', 'success')
    return redirect(url_for('admin_request'))


@app.route('/admin/request/<int:request_id>/upload', methods=['POST'])
@admin_required
def upload_recording_request(request_id):
    recording_request = RecordingRequest.query.get_or_404(request_id)
    video_file = request.files.get('video_file')
    if recording_request.status != 'approved':
        flash('Only approved requests can be fulfilled.', 'danger')
    elif not video_file or not video_file.filename or not allowed_video_file(video_file.filename):
        flash('Please select a valid video file.', 'danger')
    else:
        original_name = secure_filename(video_file.filename)
        extension = original_name.rsplit('.', 1)[1].lower()
        user_rec_dir = os.path.join(BASE_RECORDINGS_DIR, str(recording_request.user_id), 'recordings')
        os.makedirs(user_rec_dir, exist_ok=True)
        filename = f'request_{recording_request.id}.{extension}'
        video_path = os.path.join(user_rec_dir, filename)
        video_file.save(video_path)
        recording_request.status = 'fulfilled'
        recording_request.video_filename = original_name
        recording_request.video_path = video_path
        recording_request.fulfilled_at = datetime.utcnow()
        db.session.commit()
        flash('Recording request fulfilled.', 'success')
    return redirect(url_for('admin_request'))


@app.route('/admin/clear_events', methods=['POST'])
@admin_required
def admin_clear_events():
    clear_audit_events_for_user(current_user.id)
    db.session.commit()
    flash('All audit logs were cleared.', 'success')
    return redirect(url_for('admin_logs'))


@app.route('/admin/system')
@admin_required
def admin_system():
    settings = Settings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = Settings(user_id=current_user.id)
        db.session.add(settings)
        db.session.commit()

    return render_template(
        'admin_system.html',
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
    record_audit_event(current_user.id, 'settings', 'Updated system settings', 'Admin', get_client_ip())

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
    clear_audit_events_for_user(current_user.id)
    db.session.commit()
    flash('All audit logs were cleared.', 'success')
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

    new_cam = Camera(
        user_id=current_user.id,
        source=source,
        name=name,
        is_public=is_admin_user(current_user) or request.form.get('is_public') == 'true',
    )
    db.session.add(new_cam)
    db.session.commit()
    acquire_stream(current_user.id, new_cam)
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
        manager = active_user_streams.get(user_key, {}).get(camera.id)
    if manager:
        release_stream(user_key, camera.id, manager.scope_key, manager)
    
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
    if camera.user_id != current_user.id and not camera.is_public:
        return "Unauthorized", 403

    stream_user_id = camera.user_id if camera.is_public else current_user.id
    return Response(gen_frames(stream_user_id, camera), mimetype='multipart/x-mixed-replace; boundary=frame')


def camera_status(camera):
    """Get live capture health for a camera without starting a new stream."""
    stream_owner_id = camera.user_id
    with stream_lock:
        manager = active_user_streams.get(stream_owner_id, {}).get(camera.id)
        if manager is None:
            manager = active_physical_streams.get(stream_scope_key(stream_owner_id, camera))
    return bool(manager and manager.is_connected()), bool(manager and manager.is_recording)


@app.route('/api/camera-status')
@login_required
def camera_status_api():
    if is_admin_user(current_user):
        cameras = Camera.query.order_by(Camera.id).all()
    else:
        cameras = Camera.query.filter(
            or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
        ).order_by(Camera.id).all()
    statuses = {}
    for camera in cameras:
        connected, recording = camera_status(camera)
        statuses[str(camera.id)] = {
            'connected': connected,
            'recording': recording,
        }
    connected_count = sum(status['connected'] for status in statuses.values())
    return jsonify({
        'cameras': statuses,
        'connected_count': connected_count,
        'total_count': len(statuses),
    })


def acquire_stream(user_id, camera):
    """Subscribe a user to the one worker serving this camera source."""
    source_key = stream_scope_key(user_id, camera)
    with stream_lock:
        manager = active_physical_streams.get(source_key)
        if manager is None or manager.stream_stop_event.is_set():
            with app.app_context():
                user_settings = Settings.query.filter_by(user_id=user_id).first()
            manager = VideoStreamManager(user_id, camera.id, camera.source, user_settings, camera.name)
            manager.scope_key = source_key
            active_physical_streams[source_key] = manager
            manager.start()

        manager.subscriber_count += 1
        active_user_streams.setdefault(user_id, {})[camera.id] = manager
        return source_key, manager


def release_stream(user_id, camera_id, source_key, manager):
    """Remove one subscriber and stop the worker after the last subscriber leaves."""
    with stream_lock:
        user_streams = active_user_streams.get(user_id)
        if user_streams and user_streams.get(camera_id) is manager:
            del user_streams[camera_id]
            if not user_streams:
                del active_user_streams[user_id]

        manager.subscriber_count = max(0, manager.subscriber_count - 1)
        if manager.subscriber_count == 0:
            if active_physical_streams.get(source_key) is manager:
                del active_physical_streams[source_key]
            manager.release()


def gen_frames(user_id, camera):
    """Subscribe to the shared source worker and stream its output only when updated."""
    source_key, manager = acquire_stream(user_id, camera)
    last_seen_id = -1

    try:
        while not manager.stream_stop_event.is_set():
            frame_id, frame = manager.get_processed_frame(last_seen_id)
            if frame is not None:
                last_seen_id = frame_id
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                del frame
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.03)  # Rate limit streaming checks to ~30 FPS max
    finally:
        release_stream(user_id, camera.id, source_key, manager)

@app.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    user_settings = Settings.query.filter_by(user_id=current_user.id).first()
    
    user_faces_dir = os.path.join(BASE_RECORDINGS_DIR, str(current_user.id), "known_faces")
    known_faces_list = []
    if os.path.exists(user_faces_dir):
        known_faces_list = [name for name in os.listdir(user_faces_dir) if os.path.isdir(os.path.join(user_faces_dir, name))]

    if request.method == 'POST':
        user_settings.yolo_enabled = 'yolo_enabled' in request.form
        user_settings.face_recognition_enabled = 'face_recognition_enabled' in request.form
        try:
            user_settings.frame_process_interval = max(1, min(30, int(request.form.get('frame_process_interval', 3))))
            user_settings.object_detection_confidence = max(0.1, min(1.0, float(request.form.get('object_detection_confidence', 0.5))))
            user_settings.face_recognition_confidence = max(0.1, min(1.0, float(request.form.get('face_recognition_confidence', 0.6))))
        except (TypeError, ValueError):
            flash('Detection settings must contain valid numeric values.', 'danger')
            return redirect(url_for('settings'))
        db.session.commit()
        flash("Settings Updated", "success")
        
        with stream_lock:
            if current_user.id in active_user_streams:
                for mgr in active_user_streams[current_user.id].values():
                    mgr.update_settings(user_settings)
        
        return redirect(url_for('settings'))
        
    return render_template('settings.html', settings=user_settings, known_faces=known_faces_list)


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()

    if not username or not email:
        flash('Username and email are required.', 'danger')
        return redirect(url_for('settings'))

    username_taken = User.query.filter(User.username == username, User.id != current_user.id).first()
    email_taken = User.query.filter(User.email == email, User.id != current_user.id).first()
    if username_taken or email_taken:
        flash('That username or email is already in use.', 'danger')
        return redirect(url_for('settings'))

    current_user.username = username
    current_user.email = email
    db.session.commit()
    flash('Profile updated.', 'success')
    return redirect(url_for('settings'))

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
    events = EventLog.query.filter(
        EventLog.user_id == current_user.id,
        ~EventLog.event_type.in_(list(AUDIT_EVENT_TYPES)),
    ).order_by(EventLog.timestamp.desc()).all()
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
    event = EventLog.query.filter(
        EventLog.id == event_id,
        EventLog.user_id == current_user.id,
        ~EventLog.event_type.in_(list(AUDIT_EVENT_TYPES)),
    ).first()
    if event is None:
        return jsonify({"error": "Event not found"}), 404
    
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
        EventLog.query.filter(
            EventLog.user_id == current_user.id,
            ~EventLog.event_type.in_(list(AUDIT_EVENT_TYPES)),
        ).delete(synchronize_session=False)
        db.session.commit()
        return jsonify({"success": True, "message": "All security events cleared"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- INITIALIZATION ---
def init_app():
    global yolo_model, yolo_model_object, detection_pool, ALL_YOLO_CLASS_NAMES

    with app.app_context():
        ensure_database_schema()

    if yolo_model is None:
        yolo_model = YOLO(MODEL_PATH)
        if yolo_model.names:
            ALL_YOLO_CLASS_NAMES = list(yolo_model.names.values())

    if yolo_model_object is None:
        yolo_model_object = YOLO(MODEL_PATH_OBJECT)

    if detection_pool is None:
        detection_pool = ThreadPoolExecutor(max_workers=MAX_DETECTION_POOL_WORKERS)

    with app.app_context():
        for camera in Camera.query.filter_by(is_active=True).all():
            acquire_stream(camera.user_id, camera)

    print("App Initialized.")


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
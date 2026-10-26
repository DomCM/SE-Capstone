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
import json
import hashlib
import secrets
import gc
import math
import re
import stat
import torch
from urllib.parse import urlsplit, urlunsplit
from importlib.metadata import PackageNotFoundError, version as package_version
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
from models import db, User, Camera, Settings, DetectionThresholdSettings, SystemSettings, OtpChallenge, DetectionEvent, AuditLog, RecordingRequest
from concealment_detection import (
    BoundedConcealmentScanner,
    DEFAULT_MODEL as DEFAULT_CONCEALMENT_MODEL,
    PersonCueResult,
    parse_optional_threshold,
)
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
from email.utils import formataddr
from email.mime.image import MIMEImage
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from sqlalchemy import inspect, text, or_, func
from sqlalchemy.exc import SQLAlchemyError
from cryptography.fernet import Fernet, InvalidToken
import pyotp
import qrcode
from concurrent.futures import ThreadPoolExecutor
from performance_metrics import PerformanceMetrics
from behavior_analysis import (
    AdvancedTrajectoryTracker,
    BehaviorEvaluator,
    UnattendedObjectTracker,
)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
STORAGE_ROOT = os.path.join(APP_ROOT, 'storage')
INSTANCE_DIR = os.path.join(STORAGE_ROOT, 'instance')
BASE_RECORDINGS_DIR = os.path.join(STORAGE_ROOT, 'users_data')


def _move_storage_entry(source_path, destination_path, backup_path):
    if not os.path.exists(source_path):
        return

    if not os.path.exists(destination_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.move(source_path, destination_path)
        return

    if os.path.isdir(source_path) and os.path.isdir(destination_path):
        for entry in os.scandir(source_path):
            _move_storage_entry(
                entry.path,
                os.path.join(destination_path, entry.name),
                os.path.join(backup_path, entry.name),
            )
        os.rmdir(source_path)
        return

    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    conflict_path = backup_path
    suffix = 1
    while os.path.exists(conflict_path):
        conflict_path = f'{backup_path}.legacy-{suffix}'
        suffix += 1
    shutil.move(source_path, conflict_path)
    logging.warning(
        'Preserved conflicting legacy app data at %s instead of overwriting %s',
        conflict_path,
        destination_path,
    )


os.makedirs(STORAGE_ROOT, exist_ok=True)
_move_storage_entry(
    os.path.join(APP_ROOT, 'instance'),
    INSTANCE_DIR,
    os.path.join(STORAGE_ROOT, 'legacy_backup', 'instance'),
)
_move_storage_entry(
    os.path.join(APP_ROOT, 'users_data'),
    BASE_RECORDINGS_DIR,
    os.path.join(STORAGE_ROOT, 'legacy_backup', 'users_data'),
)
_move_storage_entry(
    os.path.join(APP_ROOT, 'smart_security.db'),
    os.path.join(INSTANCE_DIR, 'smart_security.db'),
    os.path.join(STORAGE_ROOT, 'legacy_backup', 'root', 'smart_security.db'),
)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(BASE_RECORDINGS_DIR, exist_ok=True)

app = Flask(__name__, instance_path=INSTANCE_DIR)
MANILA_TIMEZONE = ZoneInfo('Asia/Manila')
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY') or os.urandom(24)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///smart_security.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SMTP_SERVER'] = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
app.config['SMTP_PORT'] = int(os.environ.get('SMTP_PORT', '587'))
app.config['SMTP_USERNAME'] = os.environ.get('SMTP_USERNAME', '')
app.config['SMTP_PASSWORD'] = os.environ.get('SMTP_PASSWORD', '')
app.config['SMTP_FROM_NAME'] = os.environ.get('SMTP_FROM_NAME', 'Home Detection Security').strip()
app.config['PUBLIC_BASE_URL'] = os.environ.get('PUBLIC_BASE_URL', '').rstrip('/')
app.config['PERFORMANCE_SLOW_REQUEST_MS'] = 500
app.config['CONCEALMENT_CUE_THRESHOLD'] = parse_optional_threshold(
    os.environ.get('CONCEALMENT_CUE_THRESHOLD')
)


def to_manila_time(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(MANILA_TIMEZONE)


@app.template_filter('manila_datetime')
def format_manila_datetime(value):
    local_time = to_manila_time(value)
    return local_time.strftime('%Y-%m-%d %H:%M:%S PHT') if local_time else 'N/A'


db.init_app(app)
performance_metrics = PerformanceMetrics()
if app.config['CONCEALMENT_CUE_THRESHOLD'] is None:
    app.logger.info(
        'Background concealment cue scans are disabled; set '
        'CONCEALMENT_CUE_THRESHOLD after validating the threshold on sample data.'
    )
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

MODEL_PATH = 'models/best.pt' # main
MODEL_PATH_OBJECT = 'models/yolo26m.pt'  # secondary
BEHAVIOR_POSE_MODEL_PATH = os.environ.get('BEHAVIOR_POSE_MODEL')
APP_STARTED_MONOTONIC = time.monotonic()
OVERLAP_PIXELS = 44
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'webm'}
CROWD_PERSON_THRESHOLD = 3

security.configure(
    app,
    db,
    (User, Camera, Settings, OtpChallenge, DetectionEvent, AuditLog, RecordingRequest),
    BASE_RECORDINGS_DIR,
)

detection_pool = None
yolo_model = None
yolo_model_object = None
behavior_pose_model = None
ALL_YOLO_CLASS_NAMES = []

active_user_streams = {}
active_physical_streams = {}
stream_lock = threading.Lock()
STREAM_HEALTH_TIMEOUT_SECONDS = 3

face_cache = {}
MAX_DETECTION_POOL_WORKERS = max(1, min(2, cpu_count() or 1))
PROCESSING_INTERVAL_SECONDS = 0.1
detection_pool = None
face_recognition_lock = threading.Lock()
yolo_model_lock = threading.Lock()
yolo_model_object_lock = threading.Lock()
behavior_pose_model_lock = threading.Lock()
email_alert_lock = threading.Lock()
concealment_scan_service = BoundedConcealmentScanner(
    model_path=os.environ.get('CONCEALMENT_CUE_MODEL') or DEFAULT_CONCEALMENT_MODEL,
    max_queue_size=8,
)
RETENTION_CLEANUP_INTERVAL_SECONDS = 24 * 60 * 60
_retention_cleanup_thread = None
_retention_cleanup_wakeup = threading.Event()


def predict_with_model_lock(model, model_lock, frame, **kwargs):
    """Serialize calls to a shared Ultralytics model and its lazy predictor."""
    with model_lock:
        return model.predict(frame, **kwargs)


def get_detection_threshold_values(user_id, fallback=0.5):
    thresholds = DetectionThresholdSettings.query.filter_by(user_id=user_id).first()
    if thresholds:
        return thresholds.fire_confidence, thresholds.smoke_confidence
    return fallback, fallback


def get_or_create_detection_threshold_settings(user_id, fallback=0.5):
    thresholds = DetectionThresholdSettings.query.filter_by(user_id=user_id).first()
    if thresholds is None:
        thresholds = DetectionThresholdSettings(
            user_id=user_id,
            fire_confidence=fallback,
            smoke_confidence=fallback,
        )
        db.session.add(thresholds)
    return thresholds


def parse_confidence_threshold(value, field_name):
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{field_name} must be a number between 0.1 and 1.0.')
    if not 0.1 <= threshold <= 1.0:
        raise ValueError(f'{field_name} must be between 0.1 and 1.0.')
    return threshold


def get_detection_class_threshold(class_name, fire_confidence, smoke_confidence, fallback):
    return {
        'fire': fire_confidence,
        'smoke': smoke_confidence,
    }.get(str(class_name).strip().lower(), fallback)


def is_human_detection_name(name):
    """Return True for common YOLO labels that represent a person."""
    if not name:
        return False
    normalized = str(name).strip().lower()
    return normalized in {"person", "people", "persons", "human", "humans"}


VEHICLE_CLASS_NAMES = {
    "car", "cars", "truck", "trucks", "bus", "buses", "motorcycle", "motorbike", "motorcycles",
    "motorbikes", "van", "vans", "pickup", "pickups", "vehicle", "vehicles"
}


def is_vehicle_class_name(name):
    """Return True for a vehicle class used by the YOLO detector."""
    if not name:
        return False
    normalized = str(name).strip().lower()
    return normalized in VEHICLE_CLASS_NAMES


def normalize_virtual_lines(raw_lines):
    """Parse serialized virtual-line data stored per camera."""
    if not raw_lines:
        return []
    try:
        payload = json.loads(raw_lines)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, list):
        return []

    def coerce_point(point):
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            return float(point[0]), float(point[1])
        if isinstance(point, dict):
            if 'x' in point and 'y' in point:
                return float(point['x']), float(point['y'])
            if 0 in point and 1 in point:
                return float(point[0]), float(point[1])
        return None

    valid_lines = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        points = item.get('points') or []
        if len(points) < 2:
            continue

        start = coerce_point(points[0])
        end = coerce_point(points[1])
        if start is None or end is None:
            continue

        valid_lines.append({
            'id': item.get('id', f'line-{index + 1}'),
            'name': item.get('name', f'Line {index + 1}'),
            'points': [start, end],
            'color': item.get('color', '#fbbf24'),
        })
    return valid_lines


def get_camera_virtual_lines(camera_id):
    """Fetch the virtual lines for a camera from the database."""
    if not camera_id:
        return []
    with app.app_context():
        camera = Camera.query.get(camera_id)
        if camera is None:
            return []
        return normalize_virtual_lines(camera.virtual_lines)


def save_camera_virtual_lines(camera_id, lines):
    """Persist virtual lines for a camera as JSON."""
    if not camera_id:
        return False
    with app.app_context():
        camera = Camera.query.get(camera_id)
        if camera is None:
            return False
        camera.virtual_lines = json.dumps(lines or [])
        db.session.commit()
        return True


def detect_virtual_line_crossing(previous_center, current_center, line):
    """Return the crossing direction when an object crosses the line. Returns None if no crossing occurred."""
    if previous_center is None or current_center is None or not line or len(line) < 2:
        return None

    start = np.asarray(line[0], dtype=float)
    end = np.asarray(line[1], dtype=float)
    previous = np.asarray(previous_center, dtype=float)
    current = np.asarray(current_center, dtype=float)

    previous_side = np.cross(end - start, previous - start)
    current_side = np.cross(end - start, current - start)

    if previous_side == 0 or current_side == 0:
        return None
    if previous_side * current_side < 0:
        if previous_side > 0 and current_side < 0:
            return 'positive'
        if previous_side < 0 and current_side > 0:
            return 'negative'
    return None


def resolve_virtual_line_points(points, frame_width, frame_height):
    """Convert normalized line points into absolute frame coordinates when needed."""
    if not points:
        return []
    resolved = []
    for x, y in points:
        if 0.0 <= float(x) <= 1.0 and 0.0 <= float(y) <= 1.0:
            resolved.append((float(x) * frame_width, float(y) * frame_height))
        else:
            resolved.append((float(x), float(y)))
    return resolved


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


def deduplicate_human_boxes(human_boxes, overlap_threshold=0.5):
    """Merge overlapping person detections from multiple YOLO models."""
    unique_boxes = []
    for detection in sorted(human_boxes, key=lambda item: item[4], reverse=True):
        if not any(calculate_overlap_ratio(detection[:4], existing[:4]) >= overlap_threshold
                   for existing in unique_boxes):
            unique_boxes.append(detection)
    return unique_boxes


def draw_pose_skeleton(frame, keypoints, confidence_threshold=0.3):
    """Render visible COCO-17 keypoints and limbs onto a camera frame."""
    if keypoints is None or len(keypoints) < 17:
        return

    height, width = frame.shape[:2]
    joints = {}
    for index, keypoint in enumerate(keypoints[:17]):
        if len(keypoint) < 2:
            continue
        x, y = float(keypoint[0]), float(keypoint[1])
        confidence = float(keypoint[2]) if len(keypoint) > 2 else 1.0
        if (
            confidence >= confidence_threshold
            and math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x < width
            and 0 <= y < height
        ):
            joints[index] = (int(x), int(y))
            cv2.circle(frame, joints[index], 3, (255, 0, 255), -1)

    skeleton_limbs = (
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16),
    )
    for start, end in skeleton_limbs:
        if start in joints and end in joints:
            cv2.line(frame, joints[start], joints[end], (255, 255, 0), 2)


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


def get_system_settings():
    settings = db.session.get(SystemSettings, 1)
    if settings is None:
        settings = SystemSettings(id=1)
        db.session.add(settings)
        db.session.commit()
    return settings


def cleanup_expired_recordings(now=None):
    """Delete expired camera footage and fulfilled-request videos using separate policies."""
    settings = get_system_settings()
    current_time = time.time() if now is None else now
    cutoffs = {
        'camera': current_time - settings.recorded_footage_retention_days * 86400,
        'request': current_time - settings.request_video_retention_days * 86400,
    }
    deleted = {'camera': 0, 'request': 0, 'failed': 0}
    request_metadata_changed = False

    try:
        user_directories = list(os.scandir(BASE_RECORDINGS_DIR))
    except FileNotFoundError:
        return deleted
    except OSError:
        app.logger.exception('Unable to scan recordings directory %s', BASE_RECORDINGS_DIR)
        return deleted

    for user_directory in user_directories:
        if not user_directory.name.isdecimal() or not user_directory.is_dir(follow_symlinks=False):
            continue
        recordings_directory = os.path.join(user_directory.path, 'recordings')
        try:
            if not stat.S_ISDIR(os.stat(recordings_directory, follow_symlinks=False).st_mode):
                continue
        except FileNotFoundError:
            continue
        except OSError:
            app.logger.exception('Unable to inspect recordings directory %s', recordings_directory)
            deleted['failed'] += 1
            continue
        try:
            recording_files = list(os.scandir(recordings_directory))
        except FileNotFoundError:
            continue
        except OSError:
            app.logger.exception('Unable to scan recordings directory %s', recordings_directory)
            deleted['failed'] += 1
            continue

        user_id = int(user_directory.name)
        for recording_file in recording_files:
            if not recording_file.is_file(follow_symlinks=False):
                continue
            filename = recording_file.name
            request_match = re.fullmatch(r'request_(\d+)\.(?:mp4|avi|mkv|mov|webm)', filename, re.IGNORECASE)
            if request_match:
                category = 'request'
            elif re.fullmatch(r'cam_\d+_\d{8}_\d{6}\.webm', filename, re.IGNORECASE):
                category = 'camera'
            else:
                continue

            try:
                modified_at = recording_file.stat(follow_symlinks=False).st_mtime
                if modified_at > cutoffs[category]:
                    continue
                os.remove(recording_file.path)
            except OSError:
                app.logger.exception('Unable to remove expired recording %s', recording_file.path)
                deleted['failed'] += 1
                continue

            deleted[category] += 1
            if category == 'request':
                recording_request = RecordingRequest.query.filter_by(
                    id=int(request_match.group(1)),
                    user_id=user_id,
                ).first()
                if recording_request and recording_request.video_path:
                    stored_path = os.path.normcase(os.path.abspath(recording_request.video_path))
                    if stored_path == os.path.normcase(os.path.abspath(recording_file.path)):
                        recording_request.video_path = None
                        request_metadata_changed = True

    if request_metadata_changed:
        try:
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            app.logger.exception('Unable to update expired recording request metadata')
            deleted['failed'] += 1

    if deleted['camera'] or deleted['request']:
        app.logger.info(
            'Retention cleanup removed %s camera recording(s) and %s fulfilled-request video(s)',
            deleted['camera'],
            deleted['request'],
        )
    return deleted


def retention_cleanup_worker():
    while True:
        _retention_cleanup_wakeup.wait(RETENTION_CLEANUP_INTERVAL_SECONDS)
        _retention_cleanup_wakeup.clear()
        with app.app_context():
            try:
                cleanup_expired_recordings()
            except SQLAlchemyError:
                db.session.rollback()
                app.logger.exception('Retention cleanup could not load or save settings')

#main

@login_manager.user_loader
def load_user(user_id):
    user = db.session.get(User, int(user_id))
    return user if user and user.archived_at is None else None

ADMIN_EMAILS = [email.strip().lower() for email in os.environ.get('ADMIN_EMAILS', '').split(',') if email.strip()]


from security import (
    get_user_role, is_admin_user, get_client_ip, record_audit_event,
    get_dashboard_target, encrypt_totp_secret, decrypt_totp_secret,
    get_or_create_totp_secret, send_security_email,
    create_login_email_otp, create_password_reset_token,
    ensure_database_schema, create_admin_user,
)
from reports import build_report, generate_report_csv


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def safe_face_name(name):
    safe_name = secure_filename((name or '').strip())
    return safe_name[:100] if safe_name else ''


def user_faces_directory(user_id):
    return os.path.join(BASE_RECORDINGS_DIR, str(user_id), "known_faces")


def invalidate_face_cache(user_id):
    face_cache.pop(user_id, None)


def allowed_video_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in VIDEO_EXTENSIONS

def get_user_face_data(user_id):
    """Loads known faces for a specific user from disk or cache."""
    global face_cache
    if user_id in face_cache:
        return face_cache[user_id]
    
    user_faces_dir = user_faces_directory(user_id)
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
    with face_recognition_lock:
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


def get_people_without_usable_faces(roi_tasks, roi_results):
    """Return detected person boxes for which face recognition produced no usable face."""
    return [
        task[1]
        for task, results in zip(roi_tasks, roi_results)
        if not results
    ]


def person_identity_status_for_cue_scan(face_results):
    """Return why an ROI needs cue review, or None when it has a known identity."""
    if not face_results:
        return 'unavailable'
    if any(str(result[4]).strip().lower() == 'unknown' for result in face_results):
        return 'unknown'
    return None


def _build_alert_email_html(subject, body, image_frame=None):
    normalized_subject = str(subject).upper()
    if normalized_subject.startswith('CRITICAL'):
        alert_title = 'CRITICAL ALERT'
    elif 'UNIDENTIFIED' in normalized_subject:
        alert_title = 'UNIDENTIFIED PERSON DETECTED'
    else:
        alert_title = 'SECURITY ALERT'
    timestamp = datetime.utcnow().strftime('%d %b %Y • %H:%M UTC')
    return app.jinja_env.get_template('alert_email.html').render(
        subject=subject,
        alert_title=alert_title,
        alert_message=body or 'A security event was detected on your property.',
        timestamp=timestamp,
        has_image=image_frame is not None,
    )


def _build_account_status_email_html(user, approved):
    status = 'approved' if approved else 'not approved'
    message = (
        'Your registration has been approved. You can now sign in to your account.'
        if approved else
        'Your registration was not approved. Please contact your administrator if you believe this decision was made in error.'
    )
    return app.jinja_env.get_template('alert_email.html').render(
        subject=f'Home Detection Security — account {status}',
        alert_title='ACCOUNT REGISTRATION UPDATE',
        alert_message=message,
        timestamp=datetime.utcnow().strftime('%d %b %Y • %H:%M UTC'),
        has_image=False,
        account_status=True,
        account_username=user.username,
        account_status_label='Approved' if approved else 'Not approved',
    )


def send_account_status_email(user, approved):
    sender = app.config.get('SMTP_USERNAME')
    password = app.config.get('SMTP_PASSWORD')
    if not sender or not password or not user.email:
        app.logger.warning('Account status email could not be sent because SMTP or recipient settings are missing.')
        return False

    status = 'approved' if approved else 'not approved'
    subject = f'Home Detection Security — account {status}'
    body = (
        'Your registration has been approved. You can now sign in to your account.'
        if approved else
        'Your registration was not approved. Please contact your administrator if you believe this decision was made in error.'
    )
    try:
        message = MIMEMultipart('alternative')
        message['From'] = formataddr((app.config.get('SMTP_FROM_NAME', ''), sender))
        message['To'] = user.email
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain', 'utf-8'))
        message.attach(MIMEText(_build_account_status_email_html(user, approved), 'html', 'utf-8'))

        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as smtp:
            smtp.starttls()
            smtp.login(sender, password)
            smtp.send_message(message)
        app.logger.info('Account status email sent.')
        return True
    except Exception:
        app.logger.exception('Account status email delivery failed.')
        return False


def send_email_alert(
    user_settings,
    subject,
    body,
    image_frame=None,
    preference='email_alerts_enabled',
):
    if not user_settings or not getattr(user_settings, preference, False):
        app.logger.info('Security alert email suppressed: preference %s is disabled.', preference)
        return False

    sender = app.config.get('SMTP_USERNAME')
    password = app.config.get('SMTP_PASSWORD')
    recipient_email = getattr(user_settings, 'recipient_email', None)
    if not sender or not password or not recipient_email:
        app.logger.warning(
            'Security alert email could not be sent: SMTP credentials or recipient are not configured.'
        )
        return False

    try:
        html_body = _build_alert_email_html(subject, body, image_frame)
        msg = MIMEMultipart('mixed')
        msg['From'] = formataddr((app.config.get('SMTP_FROM_NAME', ''), sender))
        msg['To'] = recipient_email
        msg['Subject'] = subject

        alternative = MIMEMultipart('alternative')
        alternative.attach(MIMEText(body or 'Security alert detected.', 'plain', 'utf-8'))
        alternative.attach(MIMEText(html_body, 'html', 'utf-8'))
        msg.attach(alternative)

        if image_frame is not None:
            success, encoded_image = cv2.imencode('.jpg', image_frame)
            if success:
                image_part = MIMEImage(encoded_image.tobytes(), name='alert.jpg')
                image_part.add_header('Content-ID', '<alert-image>')
                image_part.add_header('Content-Disposition', 'inline', filename='alert.jpg')
                msg.attach(image_part)

        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as s:
            s.starttls()
            s.login(sender, password)
            s.send_message(msg)
        app.logger.info('Security alert email sent to %s', recipient_email)
        return True
    except Exception as e:
        app.logger.exception('Security alert email delivery failed: %s', e)
        return False


def _claim_email_slot(user_id, preference, timestamp_field):
    with email_alert_lock:
        with app.app_context():
            settings = Settings.query.filter_by(user_id=user_id).first()
            if not settings or not getattr(settings, preference, False) or not settings.recipient_email:
                return False

            now = datetime.utcnow()
            cooldown_minutes = max(1, settings.critical_email_cooldown_minutes or 15)
            last_sent_at = getattr(settings, timestamp_field)
            if last_sent_at and now - last_sent_at < timedelta(minutes=cooldown_minutes):
                return False

            setattr(settings, timestamp_field, now)
            db.session.commit()
            return True


def claim_critical_email_slot(user_id):
    """Claim the user's persisted fire/smoke alert cooldown before delivery."""
    return _claim_email_slot(user_id, 'email_alerts_enabled', 'last_critical_email_at')


def claim_unidentified_email_slot(user_id):
    """Claim the user's persisted unidentified-detection email cooldown."""
    return _claim_email_slot(
        user_id,
        'unidentified_email_alerts_enabled',
        'last_unidentified_email_at',
    )

#video
class VideoStreamManager:
    def __init__(self, user_id, camera_id, source, settings, camera_name=None, camera_zone=None, camera_is_public=False):
        self.user_id = user_id
        self.camera_id = camera_id
        self.camera_name = camera_name or f'Cam {camera_id}'
        self.camera_zone = camera_zone or 'Main Entrance'
        self.camera_is_public = bool(camera_is_public)
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
        self.crowd_alert_active = False
        self.face_detections = []
        self.yolo_detections = []
        self.recording_writer = None
        self.is_recording = False
        
        # Multitracking collection structure
        self.trackers = []          
        self.tracker_active = False
        self.tracker_lost = False
        self.tracker_frame_count = 0
        
        self.known_encodings, self.known_names = face_cache.get(user_id, ([], []))
        self.debug_tracker = False
        self._recent_event_cache = {}
        self._event_log_lock = threading.Lock()
        self._concealment_track_lock = threading.Lock()
        self._concealment_track_records = {}
        self._concealment_track_sequence = 0
        self._last_full_scan_time = 0.0
        self.vehicle_tracks = []
        self.person_tracks = []
        self._virtual_line_track_sequence = 0
        self.behavior_tracker = AdvancedTrajectoryTracker()
        self.behavior_evaluator = BehaviorEvaluator()
        self.unattended_object_tracker = UnattendedObjectTracker()
        self.pose_detections = []

    def start(self):
        """Start processing so the worker can establish the source connection."""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.processing_thread = threading.Thread(target=self._processing_worker, daemon=True)
            self.processing_thread.start()

    def _cache_settings(self, settings_obj):
        """Cache settings values from the SQLAlchemy object to avoid detached instance errors."""
        if settings_obj:
            object_confidence = getattr(settings_obj, 'object_detection_confidence', 0.7)
            fire_confidence, smoke_confidence = get_detection_threshold_values(
                settings_obj.user_id,
                object_confidence,
            )
            return {
                'yolo_enabled': settings_obj.yolo_enabled,
                'yolo_object_enabled': settings_obj.yolo_object_enabled,
                'face_recognition_enabled': settings_obj.face_recognition_enabled,
                'confidence_threshold': settings_obj.confidence_threshold,
                'object_detection_confidence': object_confidence,
                'fire_confidence': fire_confidence,
                'smoke_confidence': smoke_confidence,
                'face_recognition_confidence': getattr(settings_obj, 'face_recognition_confidence', 0.6),
                'active_classes': settings_obj.active_classes,
                'email_alerts_enabled': settings_obj.email_alerts_enabled,
                'unidentified_email_alerts_enabled': settings_obj.unidentified_email_alerts_enabled,
                'recipient_email': settings_obj.recipient_email,
                'critical_email_cooldown_minutes': getattr(settings_obj, 'critical_email_cooldown_minutes', 15),
                'scale_down_amount': settings_obj.scale_down_amount,
                'frame_process_interval': settings_obj.frame_process_interval,
            }
        return {}

    def update_settings(self, settings_obj):
        """Update the settings cache with new values."""
        self.settings_cache = self._cache_settings(settings_obj)
        if settings_obj and settings_obj.user_id != self.user_id:
            owner_settings = Settings.query.filter_by(user_id=self.user_id).first()
            fallback = (
                getattr(owner_settings, 'object_detection_confidence', 0.5)
                if owner_settings else 0.5
            )
            fire_confidence, smoke_confidence = get_detection_threshold_values(
                self.user_id,
                fallback,
            )
            self.settings_cache['fire_confidence'] = fire_confidence
            self.settings_cache['smoke_confidence'] = smoke_confidence

    def _get_concealment_track_id(self, bbox):
        now = time.monotonic()
        x1, y1, x2, y2 = bbox
        with self._concealment_track_lock:
            expired = [
                track_id
                for track_id, record in self._concealment_track_records.items()
                if now - record['last_seen'] > 120
            ]
            for track_id in expired:
                del self._concealment_track_records[track_id]

            best_track_id = None
            best_iou = 0.0
            for track_id, record in self._concealment_track_records.items():
                if record['last_seen_frame'] == self.frame_count:
                    continue
                old_x1, old_y1, old_x2, old_y2 = record['bbox']
                intersection_width = max(0, min(x2, old_x2) - max(x1, old_x1))
                intersection_height = max(0, min(y2, old_y2) - max(y1, old_y1))
                intersection = intersection_width * intersection_height
                current_area = max(0, x2 - x1) * max(0, y2 - y1)
                old_area = max(0, old_x2 - old_x1) * max(0, old_y2 - old_y1)
                union = current_area + old_area - intersection
                iou = intersection / union if union else 0.0
                if iou > best_iou:
                    best_track_id = track_id
                    best_iou = iou

            if best_track_id is None or best_iou < 0.35:
                self._concealment_track_sequence += 1
                best_track_id = self._concealment_track_sequence
            self._concealment_track_records[best_track_id] = {
                'bbox': tuple(bbox),
                'last_seen': now,
                'last_seen_frame': self.frame_count,
            }
            return best_track_id

    def _send_unidentified_person_email(self, image_frame, subject, body):
        if self.camera_is_public:
            concealment_scan_service.record_notification(
                'suppressed',
                'public_camera',
                camera_id=self.camera_id,
            )
            app.logger.info(
                'Unidentified-person alert email suppressed for public camera %s.',
                self.camera_id,
            )
            return False
        if not self.settings_cache.get('unidentified_email_alerts_enabled', False):
            concealment_scan_service.record_notification(
                'suppressed',
                'unidentified_email_preference_disabled',
                camera_id=self.camera_id,
            )
            app.logger.info(
                'Unidentified-person alert email suppressed for camera %s: user preference is disabled.',
                self.camera_id,
            )
            return False
        if not self.settings_cache.get('recipient_email'):
            concealment_scan_service.record_notification(
                'suppressed',
                'recipient_email_missing',
                camera_id=self.camera_id,
            )
            app.logger.warning(
                'Unidentified-person alert email suppressed for camera %s: recipient is not configured.',
                self.camera_id,
            )
            return False
        if not app.config.get('SMTP_USERNAME') or not app.config.get('SMTP_PASSWORD'):
            concealment_scan_service.record_notification(
                'suppressed',
                'smtp_credentials_missing',
                camera_id=self.camera_id,
            )
            app.logger.warning(
                'Unidentified-person alert email suppressed for camera %s: SMTP credentials are missing.',
                self.camera_id,
            )
            return False
        if not claim_unidentified_email_slot(self.user_id):
            concealment_scan_service.record_notification(
                'suppressed',
                'email_preference_changed_or_cooldown_active',
                camera_id=self.camera_id,
            )
            app.logger.info(
                'Unidentified-person alert email suppressed for camera %s: cooldown active or preference changed.',
                self.camera_id,
            )
            return False

        concealment_scan_service.record_notification('queued', camera_id=self.camera_id)
        email_settings = type(
            'EmailSettingsSnapshot',
            (),
            {**self.settings_cache, 'camera_id': self.camera_id},
        )()
        threading.Thread(
            target=self._deliver_unidentified_person_email,
            args=(email_settings, subject, body, image_frame),
            daemon=True,
        ).start()
        return True

    @staticmethod
    def _deliver_unidentified_person_email(email_settings, subject, body, image_frame):
        try:
            sent = send_email_alert(
                email_settings,
                subject,
                body,
                image_frame,
                preference='unidentified_email_alerts_enabled',
            )
            concealment_scan_service.record_notification(
                'sent' if sent else 'failed',
                camera_id=email_settings.camera_id,
            )
        except Exception:
            concealment_scan_service.record_notification(
                'failed',
                camera_id=email_settings.camera_id,
            )
            app.logger.exception('Unexpected unidentified-person email delivery error')

    def _unidentified_email_diagnostic(self):
        return {
            'preference_enabled': bool(
                self.settings_cache.get('unidentified_email_alerts_enabled', False)
            ),
            'recipient_configured': bool(self.settings_cache.get('recipient_email')),
            'smtp_username_configured': bool(app.config.get('SMTP_USERNAME')),
            'smtp_password_configured': bool(app.config.get('SMTP_PASSWORD')),
        }

    def _save_concealment_scan_artifacts(
        self,
        result,
        person_crop,
        *,
        identity_status,
        track_id,
        observed_at=None,
    ):
        scan_time = observed_at or datetime.utcnow()
        timestamp = scan_time.strftime('%Y%m%dT%H%M%S_%fZ')
        scan_directory = os.path.join(
            BASE_RECORDINGS_DIR,
            str(self.user_id),
            'image_scans',
        )
        filename_prefix = f'camera_{self.camera_id}_track_{track_id}_{timestamp}'
        original_path = os.path.join(scan_directory, f'{filename_prefix}_original.jpg')
        annotated_path = os.path.join(scan_directory, f'{filename_prefix}_annotated.jpg')
        report_path = os.path.join(scan_directory, f'{filename_prefix}_results.txt')
        annotated_crop = person_crop.copy()
        outcome = 'review required' if result.review_required else 'no combined cues'
        face_score = (
            f'{result.face_covering_score:.3f}'
            if result.face_covering_score is not None else 'not detected'
        )
        headwear_score = (
            f'{result.headwear_score:.3f}'
            if result.headwear_score is not None else 'not detected'
        )
        cv2.putText(
            annotated_crop,
            f'Cue scan: {outcome}',
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255) if result.review_required else (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated_crop,
            f'Face covering: {face_score}  Headwear: {headwear_score}',
            (8, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        report = '\n'.join((
            'Background Appearance-Cue Scan Results',
            f'Scanned at (UTC): {scan_time.isoformat()}Z',
            f'Camera: {self.camera_name} (ID {self.camera_id})',
            f'Track ID: {track_id}',
            f'Identity status: {identity_status}',
            f'Outcome: {outcome}',
            f'Face-covering score: {face_score}',
            f'Headwear score: {headwear_score}',
            f'Review threshold: {result.cue_threshold:.3f}',
            f'Cues: {", ".join(result.cues) if result.cues else "none"}',
            'Visual cues only. Human review is required for flagged results; no intent is inferred.',
            '',
        ))

        try:
            os.makedirs(scan_directory, exist_ok=True)
            for image_path, image in (
                (original_path, person_crop),
                (annotated_path, annotated_crop),
            ):
                encoded, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not encoded:
                    raise OSError(f'Unable to encode concealment scan image: {image_path}')
                temporary_path = f'{image_path}.tmp'
                with open(temporary_path, 'wb') as image_file:
                    image_file.write(buffer.tobytes())
                os.replace(temporary_path, image_path)
            temporary_report_path = f'{report_path}.tmp'
            with open(temporary_report_path, 'w', encoding='utf-8') as report_file:
                report_file.write(report)
            os.replace(temporary_report_path, report_path)
        except (OSError, cv2.error):
            app.logger.exception(
                'Unable to save concealment scan artifacts for camera %s track %s',
                self.camera_id,
                track_id,
            )

    def _handle_concealment_scan_result(
        self,
        result,
        error,
        *,
        identity_status,
        track_id,
        person_crop,
        observed_at=None,
    ):
        if error is not None:
            app.logger.error(
                'Concealment scan failed for camera %s track %s: %s',
                self.camera_id,
                track_id,
                error,
            )
            return
        if result is None:
            app.logger.error(
                'Concealment scan returned no result for camera %s track %s',
                self.camera_id,
                track_id,
            )
            return

        self._save_concealment_scan_artifacts(
            result,
            person_crop,
            identity_status=identity_status,
            track_id=track_id,
            observed_at=observed_at,
        )

        if result.review_required:
            combined_score = result.combined_score
            cue_description = 'face covering and headwear'
            event_type = 'UNIDENTIFIED PERSON - FACE COVERING + HEADWEAR'
            description = (
                'Face-covering and headwear appearance cues both met the configured '
                'review threshold. Human review is required; intent is unknown.'
            )
            metadata = {
                'camera_zone': self.camera_zone,
                'class': 'person',
                'face_status': identity_status,
                'identity_status': 'unknown_or_unavailable',
                'appearance_cues': list(result.cues),
                'cue_scores': {
                    'face_covering': result.face_covering_score,
                    'headwear': result.headwear_score,
                },
                'combined_score': combined_score,
                'cue_threshold': result.cue_threshold,
                'track_id': track_id,
            }
            if observed_at is not None:
                metadata['detected_at'] = observed_at.isoformat()
            self.log_db_event(
                event_type,
                description,
                confidence=combined_score,
                event_category='vision',
                detector='concealment',
                severity='medium',
                event_metadata=metadata,
                snapshot_frame=person_crop,
                event_timestamp=observed_at,
            )
            if not self.camera_is_public:
                self._send_unidentified_person_email(
                    person_crop,
                    'UNIDENTIFIED PERSON - MULTIPLE APPEARANCE CUES',
                    (
                        f'{cue_description.title()} appearance cues were both detected '
                        f'by {self.camera_name} in {self.camera_zone}. '
                        f'Combined score: {combined_score:.2f} '
                        f'(threshold: {result.cue_threshold:.2f}). '
                        'This is an automated visual cue only; please review the image. '
                        'No intent is inferred.'
                    ),
                )

    def _queue_concealment_scan(self, person_crop, bbox, identity_status):
        threshold = app.config.get('CONCEALMENT_CUE_THRESHOLD')
        concealment_scan_service.note_candidate(
            identity_status,
            camera_id=self.camera_id,
            threshold_disabled=threshold is None,
        )
        if threshold is None:
            return False
        track_id = self._get_concealment_track_id(bbox)
        crop_snapshot = person_crop.copy()
        observed_at = datetime.utcnow()
        return concealment_scan_service.submit(
            key=(self.camera_id, track_id),
            image=crop_snapshot,
            cue_threshold=threshold,
            callback=lambda result, error: self._handle_concealment_scan_result(
                result,
                error,
                identity_status=identity_status,
                track_id=track_id,
                person_crop=crop_snapshot,
                observed_at=observed_at,
            ),
        )

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

    def _draw_virtual_lines(self, annotated_frame, virtual_lines):
        """Render configured virtual lines on the current frame for operator awareness."""
        frame_h, frame_w = annotated_frame.shape[:2]
        for line in virtual_lines or []:
            try:
                points = resolve_virtual_line_points(line.get('points') or [], frame_w, frame_h)
                if len(points) < 2:
                    continue
                start = (int(points[0][0]), int(points[0][1]))
                end = (int(points[1][0]), int(points[1][1]))
                cv2.line(annotated_frame, start, end, (0, 255, 255), 3)
                cv2.putText(annotated_frame, line.get('name', 'Lane'), (start[0] + 8, max(16, start[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
            except Exception:
                continue

    def _update_virtual_line_events(self, detections, virtual_lines, snapshot_frame=None):
        """Check tracked people and vehicles against configured virtual lines."""
        if not virtual_lines or not detections:
            return

        current_objects = {'vehicle': [], 'person': []}
        for x1, y1, x2, y2, name, confidence in detections:
            if is_vehicle_class_name(name):
                object_type = 'vehicle'
            elif is_human_detection_name(name):
                object_type = 'person'
            else:
                continue

            current_objects[object_type].append({
                'name': name,
                'center': ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                'confidence': float(confidence) if confidence is not None else 0.0,
            })

        if not any(current_objects.values()):
            return

        now = time.monotonic()
        for object_type, objects in current_objects.items():
            tracks = self.vehicle_tracks if object_type == 'vehicle' else self.person_tracks
            detector_name = 'vehicle' if object_type == 'vehicle' else 'person'
            event_label = 'VEHICLE' if object_type == 'vehicle' else 'PERSON'

            for track in tracks:
                track['assigned'] = False

            for detected_object in objects:
                best_track = None
                best_distance = float('inf')
                for track in tracks:
                    if track.get('assigned'):
                        continue
                    distance = math.hypot(
                        detected_object['center'][0] - track['center'][0],
                        detected_object['center'][1] - track['center'][1],
                    )
                    if distance < best_distance:
                        best_track = track
                        best_distance = distance

                if best_track is None:
                    self._virtual_line_track_sequence += 1
                    tracks.append({
                        'center': detected_object['center'],
                        'last_seen': now,
                        'last_line_event_time': 0.0,
                        'assigned': True,
                        'name': detected_object['name'],
                        'track_id': self._virtual_line_track_sequence,
                    })
                    continue

                previous_center = best_track['center']
                best_track['center'] = detected_object['center']
                best_track['last_seen'] = now
                best_track['assigned'] = True
                best_track['name'] = detected_object['name']
                for line in virtual_lines:
                    line_points = resolve_virtual_line_points(
                        line.get('points') or [],
                        self.current_frame.shape[1],
                        self.current_frame.shape[0],
                    )
                    direction = detect_virtual_line_crossing(
                        previous_center,
                        detected_object['center'],
                        line_points,
                    )
                    if direction and now - best_track.get('last_line_event_time', 0.0) > 5:
                        best_track['last_line_event_time'] = now
                        event_metadata = {
                            'camera_zone': self.camera_zone,
                            'class': detected_object['name'],
                            'line_name': line.get('name', 'Lane'),
                            'direction': direction,
                            'line_points': line_points,
                        }
                        if object_type == 'person':
                            event_metadata['track_id'] = best_track['track_id']
                        self.log_db_event(
                            f"{event_label} {direction.upper()} CROSSING",
                            f"{detected_object['name']} crossed virtual line '{line.get('name', 'Lane')}'",
                            detected_object['confidence'],
                            event_category='vision',
                            detector=detector_name,
                            severity='medium',
                            event_metadata=event_metadata,
                            snapshot_frame=snapshot_frame,
                        )

            tracks[:] = [
                track for track in tracks
                if now - track.get('last_seen', now) <= 8.0
            ]

    def _finalize_processed_frame(self, annotated_frame):
        """Apply output work shared by detector and tracker frames."""
        if self.is_recording:
            if not self.recording_writer:
                self.start_recording(annotated_frame)
            self.recording_writer.write(annotated_frame)
        elif self.recording_writer:
            self.stop_recording()

        self.frame_count += 1
        return annotated_frame

    def _update_behavior_analytics(self, frame, annotated_frame):
        person_detections = deduplicate_human_boxes([
            (x1, y1, x2, y2, confidence)
            for x1, y1, x2, y2, name, _is_critical, confidence in self.yolo_detections
            if is_human_detection_name(name)
        ])
        frame_time = time.monotonic()
        track_ids = self.behavior_tracker.associate_detections(person_detections, frame_time)
        active_centroids = {}

        for detection, track_id in zip(person_detections, track_ids):
            x1, y1, x2, y2 = map(int, detection[:4])
            centroid = ((x1 + x2) / 2.0, float(y2))
            active_centroids[track_id] = centroid
            self.behavior_tracker.update(track_id, centroid, frame_time)

        for detection, track_id in zip(person_detections, track_ids):
            x1, y1, x2, y2 = map(int, detection[:4])
            speed, _heading = self.behavior_tracker.compute_kinematics(track_id)
            pacing_entropy = self.behavior_tracker.compute_pacing_entropy(track_id)
            dwell_time = self.behavior_tracker.get_dwell_time(track_id, frame_time)
            shadowing_score = self.behavior_tracker.evaluate_shadowing(track_id, active_centroids)
            spatial_anomaly = self.behavior_tracker.spatial_density.update_and_evaluate(
                active_centroids[track_id],
                frame.shape,
            )

            pose_keypoints = None
            best_overlap = 0.0
            for pose_box, keypoints in self.pose_detections:
                overlap = calculate_overlap_ratio(detection[:4], pose_box)
                if overlap > best_overlap:
                    best_overlap = overlap
                    pose_keypoints = keypoints
            if best_overlap < 0.3:
                pose_keypoints = None
            if pose_keypoints is not None:
                draw_pose_skeleton(annotated_frame, pose_keypoints)
            pose_score, pose_flags = self.behavior_evaluator.evaluate_pose_anomalies(
                pose_keypoints,
                (x1, y1, x2, y2),
            )
            instantaneous_score, flags = self.behavior_evaluator.evaluate(
                dwell_time,
                speed,
                pacing_entropy,
                False,
                pose_score,
                shadowing_score,
                spatial_anomaly,
            )
            threat_score = self.behavior_tracker.update_threat_energy(
                track_id,
                instantaneous_score,
                frame_time,
            )
            all_flags = sorted(set(flags + pose_flags))
            score_color = (
                (0, 0, 255) if threat_score >= 0.65
                else (0, 255, 255) if threat_score >= 0.35
                else (0, 255, 0)
            )
            cv2.putText(
                annotated_frame,
                f'Behavior {threat_score:.2f}',
                (x1, min(frame.shape[0] - 8, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                score_color,
                2,
            )

            if all_flags or threat_score >= 0.4:
                description = (
                    'Behavior indicators: ' + ', '.join(all_flags)
                    if all_flags else 'Elevated behavior score requires operator review.'
                )
                self.log_db_event(
                    'BEHAVIOR ANALYTIC ALERT',
                    description,
                    confidence=round(threat_score, 3),
                    event_category='vision',
                    detector='behavior',
                    severity='high' if threat_score >= 0.65 else 'medium',
                    event_metadata={
                        'camera_zone': self.camera_zone,
                        'class': 'person',
                        'track_id': track_id,
                        'behavior_score': round(threat_score, 3),
                        'instantaneous_behavior_score': round(instantaneous_score, 3),
                        'behavior_flags': all_flags,
                        'behavior_metrics': {
                            'dwell_seconds': round(dwell_time, 2),
                            'speed_pixels_per_second': round(speed, 2),
                            'pacing_entropy': round(pacing_entropy, 3),
                            'pose_score': round(pose_score, 3),
                            'shadowing_score': round(shadowing_score, 3),
                            'spatial_anomaly_score': round(spatial_anomaly, 3),
                        },
                        'pose_available': pose_keypoints is not None,
                        'zone_evaluation': 'not_configured',
                    },
                    snapshot_frame=annotated_frame,
                )

        object_detections = [
            {'box': (x1, y1, x2, y2), 'label': str(name)}
            for x1, y1, x2, y2, name, _is_critical, _confidence in self.yolo_detections
            if str(name).strip().lower() in {'backpack', 'handbag', 'suitcase'}
        ]
        unattended_alerts = self.unattended_object_tracker.evaluate_unattended_objects(
            object_detections,
            active_centroids,
            frame_time,
        )
        for alert in unattended_alerts:
            x1, y1, x2, y2 = alert['box']
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated_frame,
                f"UNATTENDED {alert['label'].upper()}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                2,
            )
            self.log_db_event(
                'UNATTENDED OBJECT ALERT',
                f"Unattended {alert['label']} remained outside the proximity threshold.",
                event_category='vision',
                detector='behavior',
                severity='medium',
                event_metadata={
                    'camera_zone': self.camera_zone,
                    'class': alert['label'],
                    'track_id': f"object-{alert['track_id']}",
                    'nearest_person_distance_pixels': alert['distance'],
                    'unattended_seconds': alert['unattended_seconds'],
                },
                snapshot_frame=annotated_frame,
            )
        self.behavior_tracker.purge_stale(frame_time)

    def _open_stream(self):
        """Lazily open the video stream with error handling, TCP forcing, and keyframe flushing."""
        attempt_started_at = time.monotonic()
        open_succeeded = False
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
                open_succeeded = True
                return True
            else:
                self.cap = None
                return False
        except Exception as e:
            print(f"Error opening stream {self.video_source}: {e}")
            self.cap = None
            return False
        finally:
            performance_metrics.record_camera_open(
                (time.monotonic() - attempt_started_at) * 1000,
                open_succeeded,
            )

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
        """Return true while the source reports recent activity, without treating short lags as a permanent disconnect."""
        cap = self.cap
        if cap is None or not cap.isOpened():
            return False
        return self.last_frame_at > 0 and time.monotonic() - self.last_frame_at <= STREAM_HEALTH_TIMEOUT_SECONDS * 2

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

            with self.frame_lock:
                frame = self.current_frame

            if frame is None and not self.is_connected():
                return None

            if frame is None:
                return None

            annotated_frame = frame.copy()
        
        scale_down = self.settings_cache.get('scale_down_amount', 2)
        process_interval = max(1, self.settings_cache.get('frame_process_interval', 3))
        active_classes = {
            item.strip().lower()
            for item in self.settings_cache.get('active_classes', 'fire,smoke').split(',')
            if item.strip()
        }
        
        global face_cache
        if self.user_id not in face_cache:
             self.known_encodings, self.known_names = get_user_face_data(self.user_id)

        # 1. KCF TRACKING FOR SHORT-LIVED ROI CONTINUATION
        if self.tracker_active:
            tracked_boxes = self._update_trackers(frame)
            
            if tracked_boxes and self.tracker_frame_count < process_interval:
                for x1, y1, x2, y2 in tracked_boxes:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
                
                for (top, right, bottom, left), name in self.face_detections:
                    color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                    cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(annotated_frame, f"Last verified: {name}", (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(annotated_frame, f"Tracking Active ({len(tracked_boxes)} Persons)", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                return self._finalize_processed_frame(annotated_frame)
            else:
                self._reset_tracker()

        # 2. FRAME SKIPPING FOR DETECTIONS
        if self.frame_count % process_interval != 0:
            return self._finalize_processed_frame(annotated_frame)

        # 3. FULL COMPUTER VISION SCAN PASS
        critical_detected = False
        detected_crit_names = []
        self.yolo_detections = []
        self.pose_detections = []
        human_boxes = []

        # Primary YOLO Model
        if self.settings_cache.get('yolo_enabled', True) and yolo_model:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            fire_conf = self.settings_cache.get('fire_confidence', obj_conf)
            smoke_conf = self.settings_cache.get('smoke_confidence', obj_conf)
            inference_started_at = time.monotonic()
            with torch.no_grad():
                try:
                    results = predict_with_model_lock(
                        yolo_model,
                        yolo_model_lock,
                        frame,
                        imgsz=1280,
                        conf=min(fire_conf, smoke_conf),
                        verbose=False,
                    )
                finally:
                    performance_metrics.record_timing(
                        'yolo_primary',
                        (time.monotonic() - inference_started_at) * 1000,
                    )
            for r in results:
                keypoints_data = (
                    r.keypoints.data.cpu().numpy()
                    if getattr(r, 'keypoints', None) is not None
                    else None
                )
                for box_index, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0].item())
                    name = r.names.get(cls_id, str(cls_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                    normalized_name = str(name).strip().lower()
                    class_threshold = get_detection_class_threshold(
                        normalized_name,
                        fire_conf,
                        smoke_conf,
                        obj_conf,
                    )
                    if confidence < class_threshold:
                        continue
                    
                    is_crit = normalized_name in active_classes
                    if is_crit:
                        critical_detected = True
                        detected_crit_names.append((name, confidence))
                    
                    self.yolo_detections.append((x1, y1, x2, y2, name, is_crit, confidence))
                    if is_human_detection_name(name):
                        human_boxes.append((x1, y1, x2, y2, confidence))
                        if keypoints_data is not None and box_index < len(keypoints_data):
                            self.pose_detections.append((
                                (x1, y1, x2, y2),
                                keypoints_data[box_index],
                            ))

        if behavior_pose_model is not None:
            inference_started_at = time.monotonic()
            with torch.no_grad():
                try:
                    pose_results = predict_with_model_lock(
                        behavior_pose_model,
                        behavior_pose_model_lock,
                        frame,
                        imgsz=1280,
                        conf=0.25,
                        verbose=False,
                    )
                finally:
                    performance_metrics.record_timing(
                        'behavior_pose',
                        (time.monotonic() - inference_started_at) * 1000,
                    )
            for result in pose_results:
                if getattr(result, 'keypoints', None) is None:
                    continue
                pose_keypoints = result.keypoints.data.cpu().numpy()
                for box_index, box in enumerate(result.boxes):
                    cls_id = int(box.cls[0].item())
                    if not is_human_detection_name(result.names.get(cls_id, str(cls_id))):
                        continue
                    if box_index < len(pose_keypoints):
                        pose_box = tuple(int(value) for value in box.xyxy[0].tolist())
                        self.pose_detections.append((pose_box, pose_keypoints[box_index]))

        # Secondary YOLO Model
        if self.settings_cache.get('yolo_object_enabled', True) and yolo_model_object:
            obj_conf = self.settings_cache.get('object_detection_confidence', 0.5)
            inference_started_at = time.monotonic()
            with torch.no_grad():
                try:
                    results = predict_with_model_lock(
                        yolo_model_object,
                        yolo_model_object_lock,
                        frame,
                        imgsz=1280,
                        conf=obj_conf,
                        verbose=False,
                    )
                finally:
                    performance_metrics.record_timing(
                        'yolo_secondary',
                        (time.monotonic() - inference_started_at) * 1000,
                    )
            for r in results:
                for box in r.boxes:
                    name = r.names.get(int(box.cls[0].item()))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0].item()) if getattr(box, 'conf', None) is not None and len(box.conf) > 0 else obj_conf
                    self.yolo_detections.append((x1, y1, x2, y2, name, False, confidence))
                    if is_human_detection_name(name):
                        human_boxes.append((x1, y1, x2, y2, confidence))

        # Draw YOLO detections
        human_boxes = deduplicate_human_boxes(human_boxes)

        for detection in self.yolo_detections:
            x1, y1, x2, y2, name, is_crit, confidence = detection
            color = (0, 0, 255) if is_crit else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        try:
            virtual_lines = get_camera_virtual_lines(self.camera_id)
        except Exception:
            virtual_lines = []
        self._draw_virtual_lines(annotated_frame, virtual_lines)
        vehicle_detections = [
            (x1, y1, x2, y2, name, confidence)
            for x1, y1, x2, y2, name, is_crit, confidence in self.yolo_detections
            if is_vehicle_class_name(name)
        ]
        detected_person_boxes = [
            (x1, y1, x2, y2, confidence)
            for x1, y1, x2, y2, name, is_crit, confidence in self.yolo_detections
            if is_human_detection_name(name)
        ]
        person_detections = [
            (x1, y1, x2, y2, 'person', confidence)
            for x1, y1, x2, y2, confidence in deduplicate_human_boxes(detected_person_boxes)
        ]
        try:
            self._update_virtual_line_events(
                vehicle_detections + person_detections,
                virtual_lines,
                annotated_frame,
            )
        except Exception:
            pass
        self._update_behavior_analytics(frame, annotated_frame)

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
                inference_started_at = time.monotonic()
                try:
                    all_results = run_roi_detection(roi_tasks)
                finally:
                    performance_metrics.record_timing(
                        'face_recognition',
                        (time.monotonic() - inference_started_at) * 1000,
                    )

                for roi_task, face_results in zip(roi_tasks, all_results):
                    identity_status = person_identity_status_for_cue_scan(face_results)
                    if identity_status is None:
                        continue
                    self._queue_concealment_scan(
                        roi_task[0],
                        roi_task[1],
                        identity_status,
                    )

                for bbox in get_people_without_usable_faces(roi_tasks, all_results):
                    self.log_db_event(
                        'PERSON DETECTED - FACE NOT VISIBLE OR USABLE',
                        'A person was detected, but no usable face was available for recognition.',
                        event_category='vision',
                        detector='face',
                        severity='medium',
                        event_metadata={
                            'camera_zone': self.camera_zone,
                            'class': 'person',
                            'face_status': 'unavailable',
                        },
                        snapshot_frame=annotated_frame,
                    )
                    camera_label = self.camera_name or f'Camera {self.camera_id}'
                    self._send_unidentified_person_email(
                        annotated_frame,
                        'UNIDENTIFIED PERSON - FACE NOT USABLE',
                        (
                            f'A person was detected by {camera_label} in {self.camera_zone}, '
                            'but face recognition could not assess a usable face. '
                            'Distance, pose, lighting, occlusion, or camera resolution '
                            'may explain this; please review the camera image.'
                        ),
                    )

                self.face_detections = []
                for roi_index, results_list in enumerate(all_results):
                    for t, r, b, l, name in results_list:
                        self.face_detections.append(((t, r, b, l), name))

                        if name == "Unknown":
                            self.log_db_event(
                                "UNKNOWN FACE",
                                "Unidentified person detected in an ROI.",
                                confidence=face_conf,
                                event_category='vision',
                                detector='face',
                                severity='high',
                                event_metadata={'camera_zone': self.camera_zone, 'class': 'person'},
                                snapshot_frame=annotated_frame,
                            )
                            camera_label = self.camera_name or f'Camera {self.camera_id}'
                            self._send_unidentified_person_email(
                                annotated_frame,
                                'UNIDENTIFIED PERSON DETECTED',
                                (
                                    f'An unidentified person was detected by {camera_label} '
                                    f'in {self.camera_zone}.'
                                ),
                            )
                        else:
                            self.log_db_event(
                                "RECOGNIZED",
                                f"Identified {name}",
                                confidence=face_conf,
                                event_category='vision',
                                detector='face',
                                severity='normal',
                                event_metadata={'camera_zone': self.camera_zone, 'class': 'person', 'person_name': name},
                                snapshot_frame=annotated_frame,
                            )

            except Exception as e:
                print(f"Parallel Face Recognition Error: {e}")

            # Draw startup face bounds
            for (top, right, bottom, left), name in self.face_detections:
                color = (0, 165, 255) if name == "Unknown" else (255, 255, 0)
                cv2.rectangle(annotated_frame, (left, top), (right, bottom), color, 2)
                cv2.putText(annotated_frame, name, (left, bottom+20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            bboxes_to_track = [box[:4] for box in human_boxes]
            self._initialize_trackers(frame, bboxes_to_track)
        elif human_boxes:
            frame_h, frame_w = frame.shape[:2]
            for x1, y1, x2, y2, _confidence in human_boxes:
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame_w, int(x2)), min(frame_h, int(y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                self._queue_concealment_scan(
                    frame[y1:y2, x1:x2],
                    (x1, y1, x2, y2),
                    'unavailable',
                )

        # Handle Alerts
        if critical_detected and not self.fire_alert_active:
            detected_names = list(dict.fromkeys(name for name, _ in detected_crit_names))
            highest_confidence = max((confidence for _, confidence in detected_crit_names), default=None)
            event_type = f"CRITICAL ALERT: {', '.join(detected_names)}"
            msg = f"Detected: {', '.join(detected_names)}"
            self.log_db_event(
                event_type,
                msg,
                highest_confidence,
                event_category='vision',
                detector='yolo',
                severity='critical',
                event_metadata={'camera_zone': self.camera_zone, 'classes': detected_names},
                snapshot_frame=annotated_frame,
            )
            class SettingsObj:
                pass
            settings_obj = SettingsObj()
            for k, v in self.settings_cache.items():
                setattr(settings_obj, k, v)
            if claim_critical_email_slot(self.user_id):
                threading.Thread(target=send_email_alert, args=(settings_obj, "CRITICAL ALERT", msg, annotated_frame), daemon=True).start()
        
        self.fire_alert_active = critical_detected

        crowd_detected = len(human_boxes) > CROWD_PERSON_THRESHOLD
        if crowd_detected and not self.crowd_alert_active:
            crowd_description = f"Crowd detected: {len(human_boxes)} people in camera view."
            crowd_confidence = max((box[4] for box in human_boxes), default=None)
            self.log_db_event(
                'CROWD DETECTED',
                crowd_description,
                crowd_confidence,
                event_category='vision',
                detector='yolo',
                severity='medium',
                event_metadata={'camera_zone': self.camera_zone, 'person_count': len(human_boxes)},
                snapshot_frame=annotated_frame,
            )

        self.crowd_alert_active = crowd_detected

        return self._finalize_processed_frame(annotated_frame)

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

    def log_db_event(self, event_type, desc, confidence=None, event_category='system', detector=None, severity=None, event_metadata=None, snapshot_frame=None, event_timestamp=None):
        track_id = event_metadata.get('track_id') if isinstance(event_metadata, dict) else None
        dedupe_key = (event_type, (desc or '')[:180], track_id)
        now = time.monotonic()
        with self._event_log_lock:
            last_logged = self._recent_event_cache.get(dedupe_key)
            if last_logged and now - last_logged < 15:
                return
            self._recent_event_cache[dedupe_key] = now
            if len(self._recent_event_cache) > 2048:
                cutoff = now - 60
                self._recent_event_cache = {
                    key: timestamp
                    for key, timestamp in self._recent_event_cache.items()
                    if timestamp >= cutoff
                }

        if not severity:
            normalized = (event_type or '').lower()
            if any(keyword in normalized for keyword in ('critical', 'unknown', 'alert', 'fire', 'smoke')):
                severity = 'high'
            elif any(keyword in normalized for keyword in ('crowd', 'motion', 'person', 'face')):
                severity = 'medium'
            else:
                severity = 'normal'

        if event_metadata is None:
            event_metadata = {'camera_zone': getattr(self, 'camera_zone', self.camera_name)}
        if isinstance(event_metadata, dict):
            event_metadata.setdefault('camera_is_public', self.camera_is_public)

        snapshot_bytes = None
        if snapshot_frame is not None:
            try:
                frame_height, frame_width = snapshot_frame.shape[:2]
                if frame_width > 1280:
                    target_height = max(1, round(frame_height * 1280 / frame_width))
                    snapshot_frame = cv2.resize(snapshot_frame, (1280, target_height), interpolation=cv2.INTER_AREA)
                encoded, image = cv2.imencode('.jpg', snapshot_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if encoded:
                    snapshot_bytes = image.tobytes()
                else:
                    app.logger.error('Unable to encode snapshot for event %s', event_type)
            except (AttributeError, cv2.error, ValueError):
                app.logger.exception('Unable to prepare snapshot for event %s', event_type)

        with app.app_context():
            try:
                event_values = {
                    'user_id': self.user_id,
                    'camera_id': self.camera_id,
                    'source_name': self.camera_name,
                    'event_type': event_type,
                    'description': desc,
                    'confidence': confidence,
                    'event_category': event_category,
                    'detector': detector,
                    'severity': severity,
                    'event_metadata': (
                        json.dumps(event_metadata)
                        if isinstance(event_metadata, dict)
                        else event_metadata
                    ),
                }
                if event_timestamp is not None:
                    event_values['timestamp'] = event_timestamp
                log = DetectionEvent(**event_values)
                db.session.add(log)
                db.session.commit()
                if snapshot_bytes:
                    snapshot_directory = os.path.join(BASE_RECORDINGS_DIR, str(self.user_id), 'event_snapshots')
                    snapshot_filename = f'{log.id}.jpg'
                    snapshot_path = os.path.join(snapshot_directory, snapshot_filename)
                    temporary_path = f'{snapshot_path}.tmp'
                    try:
                        os.makedirs(snapshot_directory, exist_ok=True)
                        with open(temporary_path, 'wb') as snapshot_file:
                            snapshot_file.write(snapshot_bytes)
                        os.replace(temporary_path, snapshot_path)
                        log.snapshot_path = snapshot_filename
                        db.session.commit()
                    except OSError:
                        if os.path.exists(temporary_path):
                            os.remove(temporary_path)
                        app.logger.exception('Unable to save snapshot for event %s', log.id)
                    except SQLAlchemyError:
                        db.session.rollback()
                        if os.path.exists(snapshot_path):
                            os.remove(snapshot_path)
                        app.logger.exception('Unable to attach snapshot to event %s', log.id)
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
    request.environ['performance_started_at'] = time.monotonic()
    allowed_endpoints = ['disclaimer', 'accept_terms', 'static']
    if request.endpoint in allowed_endpoints:
        return

    if not session.get('privacy_agreed'):
        return redirect(url_for('disclaimer'))

    enrollment_endpoints = {'security_setup', 'logout', 'static', 'disclaimer', 'accept_terms'}
    if (app.config.get('LEGACY_OTP') and current_user.is_authenticated and current_user.totp_secret
            and not current_user.totp_enabled
            and request.endpoint not in enrollment_endpoints):
        return redirect(url_for('security_setup'))


@app.after_request
def prevent_authenticated_page_caching(response):
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    started_at = request.environ.get('performance_started_at')
    if started_at is not None and request.endpoint != 'performance_metrics_api':
        duration_ms = (time.monotonic() - started_at) * 1000
        performance_metrics.record_request(
            request.endpoint,
            request.method,
            response.status_code,
            duration_ms,
        )
        if duration_ms >= app.config['PERFORMANCE_SLOW_REQUEST_MS']:
            app.logger.warning(
                'Slow request endpoint=%s method=%s status=%s duration_ms=%.2f',
                request.endpoint or 'unknown',
                request.method,
                response.status_code,
                duration_ms,
            )
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


@app.context_processor
def inject_admin_pending_request_count():
    pending_count = 0
    if current_user.is_authenticated and is_admin_user(current_user):
        pending_count = RecordingRequest.query.filter_by(status='pending').count()
    return {'admin_pending_request_count': pending_count}


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = User.query.filter_by(email=email).first()
        if user and user.archived_at is None:
            token = create_password_reset_token(user)
            reset_path = url_for('reset_password', token=token)
            public_base_url = app.config.get('PUBLIC_BASE_URL')
            reset_url = (
                f'{public_base_url}{reset_path}'
                if public_base_url
                else url_for('reset_password', token=token, _external=True)
            )
            send_security_email(
                user,
                'Reset your Home Detection Security password',
                (
                    'A password reset link was requested for your Home Detection Security account. '
                    'Use this link within 1 hour to choose a new password:\n\n'
                    f'{reset_url}\n\n'
                    'If you did not request this change, ignore this email. Your password will not change.'
                ),
                html_body=render_template(
                    'password_reset_email.html',
                    username=user.username,
                    reset_url=reset_url,
                ),
            )
        flash('If an active account matches that email, password reset instructions have been sent.', 'success')
        return redirect(url_for('forgot_password'))
    return render_template('password_reset_request.html')


def find_password_reset_challenge(token):
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    return OtpChallenge.query.filter_by(
        purpose='password_reset_link', code_hash=token_hash, used_at=None
    ).first()


@app.route('/verify-otp', methods=['GET', 'POST'])
def verify_otp():
    purpose = request.args.get('purpose') or session.get('otp_purpose')
    if purpose != 'login':
        return redirect(url_for('login'))
    session['otp_purpose'] = purpose

    pending_login_user = db.session.get(User, session.get('pending_login_user_id')) if purpose == 'login' else None
    otp_email = pending_login_user.email if pending_login_user else None

    if request.method == 'POST':
        code = request.form.get('otp', '').strip()
        if not code.isdigit() or len(code) != 6:
            flash('Enter the 6-digit verification code.', 'danger')
            return render_template('verify_otp.html', purpose=purpose,
                                   email=otp_email,
                                   legacy_otp=app.config.get('LEGACY_OTP', False))

        user = pending_login_user
        if app.config.get('LEGACY_OTP'):
            secret = decrypt_totp_secret(user.totp_secret) if user else None
            valid = bool(user and user.totp_enabled and secret and pyotp.TOTP(secret).verify(code))
        else:
            challenge = OtpChallenge.query.filter_by(
                user_id=user.id if user else 0, purpose='login', used_at=None
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

        if valid and user and user.archived_at is None and user.is_approved:
            user.last_login = datetime.utcnow()
            db.session.commit()
            login_user(user)
            session.pop('pending_login_user_id', None)
            session.pop('otp_purpose', None)
            return redirect(url_for(get_dashboard_target(user)))

        flash('That verification code is invalid or expired.', 'danger')

    return render_template('verify_otp.html', purpose=purpose,
                           email=otp_email,
                           legacy_otp=app.config.get('LEGACY_OTP', False))


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    token = (request.values.get('token') or '').strip()
    challenge = find_password_reset_challenge(token)
    if not challenge or challenge.expires_at <= datetime.utcnow():
        flash('That password reset link is invalid or expired. Request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    user = db.session.get(User, challenge.user_id)
    if not user or user.archived_at is not None:
        flash('That password reset link is invalid or expired. Request a new one.', 'danger')
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        password = request.form.get('password', '')
        confirmation = request.form.get('confirm_password', '')
        if len(password) < 12 or password != confirmation:
            flash('Use a strong password and make sure both fields match.', 'danger')
            return render_template('reset_password.html', token=token)
        user.password = generate_password_hash(password)
        challenge.used_at = datetime.utcnow()
        db.session.commit()
        return redirect(url_for('password_success'))
    return render_template('reset_password.html', token=token)


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
        building_number = (request.form.get('building_number') or '').strip()
        floor = (request.form.get('floor') or '').strip()
        street_address = (request.form.get('street_address') or '').strip()
        
        if not username or not email or not password or not building_number or not street_address:
            flash('Username, email, password, building/unit, and street/subdivision are required.', 'danger')
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
            is_approved=False,
            building_number=building_number[:150],
            floor=floor[:150] or None,
            street_address=street_address[:200],
            totp_secret=(encrypt_totp_secret(pyotp.random_base32())
                         if app.config.get('LEGACY_OTP') else None),
        )
        db.session.add(new_user)
        db.session.commit()

        default_settings = Settings(user_id=new_user.id, recipient_email=email)
        db.session.add(default_settings)
        db.session.commit()

        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "known_faces"), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), "recordings"), exist_ok=True)

        record_audit_event(new_user.id, 'create', f"Created account for {new_user.username}", 'Auth', get_client_ip())

        flash('Your account was created and is awaiting admin approval. You can sign in after it has been approved.', 'success')
        return redirect(url_for('login'))
    
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

        if user and user.archived_at is None and check_password_hash(user.password, password):
            if not user.is_approved:
                if user.rejected_at is not None:
                    flash('Your account registration was not approved. Please contact your administrator.', 'danger')
                else:
                    flash('Your account is awaiting admin approval.', 'danger')
                return redirect(url_for('login'))
            if not app.config.get('LEGACY_OTP'):
                session['pending_login_user_id'] = user.id
                session['otp_purpose'] = 'login'
                code = create_login_email_otp(user)
                send_security_email(
                    user,
                    'Home Detection Security — login verification code',
                    f'Your login verification code is {code}. It expires in 3 minutes.\n\n'
                    f'If you did not request this, please contact your administrator.',
                )
                return redirect(url_for('verify_otp', purpose='login'))
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
        Camera.archived_at.is_(None),
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
    ).order_by(Camera.id).all()
    online_cameras_count = sum(1 for camera in user_cameras if camera_status(camera)[0])
    archived_cameras = Camera.query.filter_by(
        user_id=current_user.id
    ).filter(Camera.archived_at.is_not(None)).order_by(Camera.id).all()
    dashboard_metrics = get_user_dashboard_metrics(current_user.id)
    recording_requests = RecordingRequest.query.filter_by(user_id=current_user.id).order_by(RecordingRequest.created_at.desc()).all()
    events = DetectionEvent.query.filter_by(
        user_id=current_user.id,
        event_category='vision',
    ).order_by(DetectionEvent.timestamp.desc()).limit(10).all()
    
    return render_template(
        'user_dashboard.html',
        cameras=user_cameras,
        online_cameras_count=online_cameras_count,
        archived_cameras=archived_cameras,
        dashboard_metrics=dashboard_metrics,
        recording_requests=recording_requests,
        events=events,
        available_zones=get_available_zones(),
    )


def get_user_dashboard_metrics(user_id):
    month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    return {
        'pending_requests': RecordingRequest.query.filter_by(
            user_id=user_id,
            status='pending',
        ).count(),
        'total_detections': DetectionEvent.query.filter_by(
            user_id=user_id,
            event_category='vision',
        ).count(),
        'high_severity_alerts': DetectionEvent.query.filter(
            DetectionEvent.user_id == user_id,
            DetectionEvent.event_category == 'vision',
            DetectionEvent.timestamp >= month_start,
            DetectionEvent.severity.in_(('high', 'critical')),
        ).count(),
    }


@app.route('/api/user/dashboard-metrics')
@login_required
def user_dashboard_metrics_api():
    if is_admin_user(current_user):
        return jsonify(error='This endpoint is for user dashboards only.'), 403
    return jsonify(get_user_dashboard_metrics(current_user.id))


@app.route('/recording_requests', methods=['POST'])
@login_required
def create_recording_request():
    camera_id = request.form.get('camera_id', type=int)
    date_needed = request.form.get('date_needed', '').strip()
    start_time = request.form.get('start_time', '').strip()
    end_time = request.form.get('end_time', '').strip()
    reason = request.form.get('reason', '').strip()
    camera = Camera.query.filter(
        Camera.id == camera_id,
        Camera.archived_at.is_(None),
        Camera.is_public.is_(True),
    ).first()

    if not camera or not date_needed or not start_time or not end_time or not reason:
        flash('Camera, date, start time, end time, and reason are required.', 'danger')
        return redirect(url_for('user_request'))

    try:
        datetime.strptime(date_needed, '%Y-%m-%d')
    except ValueError:
        flash('Please provide a valid request date.', 'danger')
        return redirect(url_for('user_request'))

    try:
        parsed_start_time = datetime.strptime(start_time, '%H:%M')
        parsed_end_time = datetime.strptime(end_time, '%H:%M')
        if parsed_start_time.strftime('%H:%M') != start_time or parsed_end_time.strftime('%H:%M') != end_time:
            raise ValueError
    except ValueError:
        flash('Please provide valid start and end times.', 'danger')
        return redirect(url_for('user_request'))

    if parsed_end_time <= parsed_start_time:
        flash('End time must be later than start time.', 'danger')
        return redirect(url_for('user_request'))

    recording_request = RecordingRequest(
        user_id=current_user.id,
        camera_id=camera.id,
        camera_name=camera.name,
        date_needed=date_needed,
        time_range=f'{start_time} - {end_time}',
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


@app.route('/recording_requests/<int:request_id>/view')
@login_required
def view_recording_request(request_id):
    recording_request = RecordingRequest.query.filter_by(id=request_id, user_id=current_user.id).first_or_404()
    if recording_request.status != 'fulfilled' or not recording_request.video_path or not os.path.isfile(recording_request.video_path):
        return 'Recording is not available.', 404
    return send_file(
        recording_request.video_path,
        as_attachment=False,
        download_name=recording_request.video_filename,
        conditional=True,
    )

@app.route('/user_request')
@login_required
def user_request():
    cameras = Camera.query.filter(
        Camera.archived_at.is_(None),
        Camera.is_public.is_(True)
    ).order_by(Camera.id).all()
    requests = RecordingRequest.query.filter_by(user_id=current_user.id).order_by(
        RecordingRequest.created_at.desc()
    ).all()
    return render_template('user_request.html', cameras=cameras, requests=requests)

@app.route('/cctv')
@login_required
def cctv():
    user_cameras = []
    cameras = Camera.query.filter(
        Camera.archived_at.is_(None),
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
    ).order_by(Camera.id).all()
    for camera in cameras:
        connected, recording = camera_status(camera)
        user_cameras.append({
            'id': camera.id,
            'name': camera.name,
            'location': '',
            'status': 'online' if connected else 'offline',
            'motion_detected': False,
            'is_recording': recording,
            'is_public': camera.is_public,
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    
    return render_template('user_cctv.html', cameras=user_cameras)


@app.route('/admin')
@admin_required
def admin_dashboard():
    total_users = User.query.count()
    total_cameras = Camera.query.filter(Camera.archived_at.is_(None)).count()
    active_cameras = Camera.query.filter_by(is_active=True, archived_at=None).count()
    total_events = DetectionEvent.query.filter_by(event_category='vision').count()
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
    events_today = DetectionEvent.query.filter(
        DetectionEvent.timestamp >= since,
        DetectionEvent.event_category == 'vision',
    ).count()
    new_users_today = User.query.filter(User.created_at >= since).count()

    alert_events_today = DetectionEvent.query.filter(
        DetectionEvent.timestamp >= since,
        DetectionEvent.event_category == 'vision',
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
    recent_events = DetectionEvent.query.order_by(
        DetectionEvent.timestamp.desc()
    ).limit(5).all()
    recent_audit_logs = AuditLog.query.order_by(
        AuditLog.timestamp.desc()
    ).limit(5).all()
    for event in sorted(
        recent_events + recent_audit_logs,
        key=lambda item: item.timestamp or datetime.min,
        reverse=True,
    )[:5]:
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


@app.route('/api/admin/pending-requests')
@admin_required
def admin_pending_requests_count():
    return jsonify(
        pending_requests_count=RecordingRequest.query.filter_by(status='pending').count()
    )


def get_available_zones():
    zones = []
    try:
        for c in Camera.query.filter(Camera.archived_at.is_(None)).all():
            z = (getattr(c, 'zone', None) or '').strip()
            if z and z not in zones:
                zones.append(z)
    except Exception:
        pass
    if not zones:
        zones = ['Main Entrance', 'North Perimeter', 'Clubhouse & Amenities']
    return zones


def format_bytes(size_bytes):
    value = float(size_bytes)
    for unit in ('B', 'KB', 'MB', 'GB', 'TB', 'PB'):
        if value < 1024 or unit == 'PB':
            return f'{value:.1f} {unit}'
        value /= 1024


def format_app_uptime(seconds):
    total_minutes = max(0, int(seconds // 60))
    days, remaining_minutes = divmod(total_minutes, 24 * 60)
    hours, minutes = divmod(remaining_minutes, 60)
    return f'{days}d {hours}h {minutes}m'


@app.route('/admin/cctv')
@admin_required
def admin_cctv():
    show_archived = request.args.get('include_archived') == '1'
    camera_items = []
    cameras_query = Camera.query
    if show_archived:
        cameras_query = cameras_query.filter(Camera.archived_at.is_not(None))
    else:
        cameras_query = cameras_query.filter(
            Camera.archived_at.is_(None),
            or_(
                Camera.user_id == current_user.id,
                Camera.is_public.is_(True),
            ),
        )
    for camera in cameras_query.order_by(Camera.id).all():
        cam_zone = getattr(camera, 'zone', 'Main Entrance') or 'Main Entrance'
        camera_items.append({
            'id': camera.id,
            'name': camera.name,
            'zone': cam_zone,
            'location': cam_zone,
            'status': 'online' if camera.is_active else 'offline',
            'motion_detected': False,
            'is_recording': False,
            'is_public': camera.is_public,
            'is_archived': camera.archived_at is not None,
            'stream_url': url_for('video_feed', camera_id=camera.id),
        })
    return render_template(
        'admin_cctv.html',
        cameras=camera_items,
        available_zones=get_available_zones(),
        show_archived=show_archived,
    )


@app.route('/admin/users')
@admin_required
def admin_users():
    include_archived = request.args.get('include_archived') == '1'
    users_query = User.query
    if not include_archived:
        users_query = users_query.filter_by(archived_at=None)
    users = users_query.order_by(User.id).all()
    return render_template('admin_users.html', users=users, include_archived=include_archived, message=None)


def _admin_user_action_response(message, notification_sent=None, status=200):
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify(success=status < 400, message=message, notification_sent=notification_sent), status
    flash(message, 'success' if status < 400 else 'danger')
    return redirect(url_for('admin_users'))


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


@app.route('/admin/users/<int:user_id>/approve', methods=['POST'])
@admin_required
def admin_approve_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.archived_at is not None:
        return _admin_user_action_response('Archived accounts cannot be approved.', status=400)
    if user.rejected_at is not None:
        return _admin_user_action_response('Rejected accounts cannot be approved.', status=400)
    if user.is_approved:
        return _admin_user_action_response(f'Account for {user.username} is already approved.', status=400)

    user.is_approved = True
    db.session.commit()
    record_audit_event(current_user.id, 'approve', f"Approved account for {user.username}", 'Admin', get_client_ip())
    notification_sent = send_account_status_email(user, approved=True)
    message = f'Account for {user.username} approved.'
    if not notification_sent:
        message += ' The account status email could not be sent; check SMTP configuration and logs.'
    return _admin_user_action_response(message, notification_sent=notification_sent)


@app.route('/admin/users/<int:user_id>/reject', methods=['POST'])
@admin_required
def admin_reject_user(user_id):
    user = User.query.get_or_404(user_id)
    if user.archived_at is not None:
        return _admin_user_action_response('Archived accounts cannot be rejected.', status=400)
    if user.is_approved:
        return _admin_user_action_response('Approved accounts cannot be rejected.', status=400)
    if user.rejected_at is not None:
        return _admin_user_action_response(f'Account for {user.username} was already rejected.', status=400)

    user.rejected_at = datetime.utcnow()
    db.session.commit()
    record_audit_event(current_user.id, 'deny', f"Rejected account for {user.username}", 'Admin', get_client_ip())
    notification_sent = send_account_status_email(user, approved=False)
    message = f'Account for {user.username} rejected.'
    if not notification_sent:
        message += ' The account status email could not be sent; check SMTP configuration and logs.'
    return _admin_user_action_response(message, notification_sent=notification_sent)


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
        user.archived_at = datetime.utcnow()
        db.session.commit()
        record_audit_event(current_user.id, 'archive', f"Archived user \"{user.username}\"", 'Admin', get_client_ip())
        flash(f'User {user.username} archived.', 'success')
    except Exception as exc:
        db.session.rollback()
        flash(f'Unable to delete user: {exc}', 'danger')
    return redirect(url_for('admin_users'))


@app.route('/admin/users/<int:user_id>/restore', methods=['POST'])
@admin_required
def admin_restore_user(user_id):
    if user_id == current_user.id:
        flash('You cannot restore your own account.', 'danger')
        return redirect(url_for('admin_users'))

    user = User.query.get_or_404(user_id)
    if user.archived_at is None:
        flash('That account is already active.', 'danger')
        return redirect(url_for('admin_users'))

    user.archived_at = None
    db.session.commit()
    record_audit_event(current_user.id, 'restore', f"Restored user \"{user.username}\"", 'Admin', get_client_ip())
    flash(f'User {user.username} restored.', 'success')
    return redirect(url_for('admin_users'))


@app.route('/admin/logs')
@admin_required
def admin_logs():
    query = AuditLog.query.join(User, AuditLog.user_id == User.id, isouter=True)

    user_filter = (request.args.get('user') or '').strip()
    action_filter = (request.args.get('action') or '').strip()
    from_date = (request.args.get('from') or '').strip()
    to_date = (request.args.get('to') or '').strip()

    if user_filter:
        search = f"%{user_filter}%"
        query = query.filter(or_(User.username.ilike(search), User.email.ilike(search)))

    if action_filter:
        query = query.filter(AuditLog.event_type.ilike(action_filter))

    if from_date:
        try:
            from_dt = datetime.strptime(from_date, '%Y-%m-%d')
            query = query.filter(AuditLog.timestamp >= from_dt)
        except ValueError:
            pass

    if to_date:
        try:
            to_dt = datetime.strptime(to_date, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(AuditLog.timestamp < to_dt)
        except ValueError:
            pass

    logs = query.order_by(AuditLog.timestamp.desc()).all()
    return render_template('admin_logs.html', logs=logs, total_logs=len(logs), page=1)


@app.route('/admin/reports')
@admin_required
def admin_reports():
    report_context = build_report(
        DetectionEvent,
        report_type=request.args.get('type', 'incident'),
        timeframe=request.args.get('timeframe', '7d'),
        zone=request.args.get('zone', 'all'),
        severity=request.args.get('severity', 'all'),
        start_date=request.args.get('start_date'),
        end_date=request.args.get('end_date'),
        camera_model=Camera,
    )
    for log in report_context['incident_logs']:
        log['snapshot_url'] = (
            url_for('admin_event_snapshot', event_id=log['event_id'])
            if log['snapshot_available'] else None
        )
    return render_template('admin_reports.html', **report_context)


@app.route('/admin/reports/export')
@admin_required
def admin_reports_export():
    report_context = build_report(
        DetectionEvent,
        report_type=request.args.get('type', 'incident'),
        timeframe=request.args.get('timeframe', '7d'),
        zone=request.args.get('zone', 'all'),
        severity=request.args.get('severity', 'all'),
        start_date=request.args.get('start_date'),
        end_date=request.args.get('end_date'),
        camera_model=Camera,
    )
    csv_data = generate_report_csv(report_context)
    filename = f"security_report_{report_context['selected_type']}_{datetime.utcnow().strftime('%Y%m%d')}.csv"
    record_audit_event(
        current_user.id, 'export',
        f"Exported {report_context['selected_type']} security report",
        'Admin', get_client_ip(),
    )
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route('/admin/request')
@admin_required
def admin_request():
    requests = []
    for recording_request in RecordingRequest.query.order_by(RecordingRequest.created_at.desc()).all():
        requests.append({
            'id': recording_request.id,
            'username': recording_request.user.username,
            'camera_name': (
                recording_request.camera_name
                or (recording_request.camera.name if recording_request.camera else 'Camera removed')
            ),
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
        record_audit_event(
            current_user.id, 'approve',
            f"Approved recording request #{recording_request.id} for user {recording_request.user_id}",
            'Admin', get_client_ip(),
        )
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
        record_audit_event(
            current_user.id, 'deny',
            f"Denied recording request #{recording_request.id} for user {recording_request.user_id}",
            'Admin', get_client_ip(),
        )
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
        record_audit_event(
            current_user.id, 'fulfill',
            f"Fulfilled recording request #{recording_request.id} for user {recording_request.user_id}",
            'Admin', get_client_ip(),
        )
        flash('Recording request fulfilled.', 'success')
    return redirect(url_for('admin_request'))


@app.route('/api/camera/<int:camera_id>/virtual_lines', methods=['GET', 'POST', 'DELETE'])
@login_required
def camera_virtual_lines(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.archived_at is not None:
        return jsonify({'error': 'Camera is archived'}), 404
    if camera.user_id != current_user.id and not is_admin_user(current_user):
        return jsonify({'error': 'Unauthorized'}), 403

    if request.method == 'GET':
        return jsonify({'camera_id': camera.id, 'lines': normalize_virtual_lines(camera.virtual_lines)})

    if request.method == 'DELETE':
        payload = request.get_json(silent=True) or {}
        existing_lines = normalize_virtual_lines(camera.virtual_lines)
        line_id = payload.get('line_id')
        if payload.get('clear_all') or line_id is None:
            updated_lines = []
        else:
            updated_lines = [line for line in existing_lines if str(line.get('id')) != str(line_id)]

        camera.virtual_lines = json.dumps(updated_lines)
        db.session.commit()
        if is_admin_user(current_user):
            record_audit_event(
                current_user.id, 'update',
                f"Removed virtual line(s) for camera \"{camera.name}\" (#{camera.id})",
                'Admin', get_client_ip(),
            )
        return jsonify({'success': True, 'lines': updated_lines})

    payload = request.get_json(silent=True) or {}
    lines = payload.get('lines') or []
    if not isinstance(lines, list):
        return jsonify({'error': 'Invalid line payload'}), 400

    normalized = []
    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        points = line.get('points') or []
        if len(points) < 2:
            continue
        normalized.append({
            'id': line.get('id', f'line-{index + 1}'),
            'name': line.get('name', f'Line {index + 1}'),
            'color': line.get('color', '#facc15'),
            'points': points,
        })

    camera.virtual_lines = json.dumps(normalized)
    db.session.commit()
    if is_admin_user(current_user):
        record_audit_event(
            current_user.id, 'update',
            f"Updated virtual lines for camera \"{camera.name}\" (#{camera.id})",
            'Admin', get_client_ip(),
        )
    return jsonify({'success': True, 'lines': normalized})


@app.route('/api/camera/<int:camera_id>/snapshot')
@login_required
def camera_snapshot(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.archived_at is not None:
        return jsonify({'error': 'Camera is archived'}), 404
    if camera.user_id != current_user.id and not is_admin_user(current_user):
        return jsonify({'error': 'Unauthorized'}), 403

    stream_user_id = camera.user_id if camera.is_public else current_user.id
    source_key, manager = acquire_stream(stream_user_id, camera)
    try:
        frame = manager.process_frame()
        if frame is None:
            return jsonify({'error': 'No snapshot available yet'}), 503
        success, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not success:
            return jsonify({'error': 'Unable to encode snapshot'}), 500
        encoded_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        return jsonify({'success': True, 'image': f'data:image/jpeg;base64,{encoded_b64}'})
    finally:
        release_stream(stream_user_id, camera.id, source_key, manager)


@app.route('/admin/clear_events', methods=['POST'])
@admin_required
def admin_clear_events():
    cleared_count = clear_admin_vision_events_and_snapshots()
    db.session.commit()
    record_audit_event(
        current_user.id, 'clear',
        f'Cleared {cleared_count} admin computer-vision event log(s); audit trail preserved',
        'Admin', get_client_ip(),
    )
    flash('Computer-vision event logs for all admin accounts were cleared. User events and audit logs were preserved.', 'success')
    return redirect(url_for('admin_logs'))


@app.route('/admin/system')
@admin_required
def admin_system():
    settings = Settings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = Settings(user_id=current_user.id)
        db.session.add(settings)
        db.session.commit()
    fire_confidence, smoke_confidence = get_detection_threshold_values(
        current_user.id,
        settings.object_detection_confidence,
    )
    system_settings = get_system_settings()

    try:
        disk_usage = shutil.disk_usage(BASE_RECORDINGS_DIR)
        storage_used = format_bytes(disk_usage.used)
        storage_free = format_bytes(disk_usage.free)
        storage_total = format_bytes(disk_usage.total)
        storage_pct = round(disk_usage.used / disk_usage.total * 100, 1) if disk_usage.total else None
    except OSError:
        storage_used = storage_free = storage_total = storage_pct = None

    active_cameras = Camera.query.filter_by(is_active=True, archived_at=None).all()
    connected_cameras = sum(1 for camera in active_cameras if camera_status(camera)[0])
    try:
        flask_version = package_version('Flask')
    except PackageNotFoundError:
        flask_version = None

    return render_template(
        'admin_system.html',
        settings=settings,
        fire_confidence=fire_confidence,
        smoke_confidence=smoke_confidence,
        system_settings=system_settings,
        storage_pct=storage_pct,
        storage_used=storage_used,
        storage_free=storage_free,
        storage_total=storage_total,
        sys_info={
            'python_version': sys.version.split()[0],
            'flask_version': flask_version,
            'cv2_version': cv2.__version__,
            'uptime': format_app_uptime(time.monotonic() - APP_STARTED_MONOTONIC),
            'camera_count': f'{connected_cameras} / {len(active_cameras)} active',
            'host': request.host,
        },
        message=None,
    )


@app.route('/admin/api/performance-metrics')
@admin_required
def performance_metrics_api():
    metrics = performance_metrics.snapshot()
    with stream_lock:
        managers = list(active_physical_streams.values())

    metrics['camera_workers'] = {
        'active': len(managers),
        'connected': sum(manager.is_connected() for manager in managers),
        'reader_threads': sum(
            bool(manager.reader_thread and manager.reader_thread.is_alive())
            for manager in managers
        ),
        'processing_threads': sum(
            bool(manager.processing_thread and manager.processing_thread.is_alive())
            for manager in managers
        ),
    }
    return jsonify(metrics)


@app.route('/admin/api/concealment-status')
@admin_required
def admin_concealment_status_api():
    with stream_lock:
        managers = list(active_physical_streams.values())

    threshold = app.config.get('CONCEALMENT_CUE_THRESHOLD')
    camera_statuses = []
    for manager in managers:
        camera_statuses.append({
            'camera_id': manager.camera_id,
            'camera_name': manager.camera_name,
            'camera_is_public': manager.camera_is_public,
            'is_connected': manager.is_connected(),
            'face_recognition_enabled': bool(
                manager.settings_cache.get('face_recognition_enabled', True)
            ),
            'cue_scan_enabled': threshold is not None,
            'email': manager._unidentified_email_diagnostic(),
        })

    return jsonify({
        'enabled': threshold is not None,
        'threshold': threshold,
        'service': concealment_scan_service.status(),
        'smtp_server_configured': bool(app.config.get('SMTP_SERVER')),
        'smtp_username_configured': bool(app.config.get('SMTP_USERNAME')),
        'smtp_password_configured': bool(app.config.get('SMTP_PASSWORD')),
        'active_camera_count': len(managers),
        'cameras': camera_statuses,
    })


@app.route('/api/concealment-status')
@login_required
def user_concealment_status_api():
    cameras = Camera.query.filter_by(user_id=current_user.id).order_by(Camera.id).all()
    settings = Settings.query.filter_by(user_id=current_user.id).first()
    camera_ids = {camera.id for camera in cameras}
    with stream_lock:
        managers = {
            manager.camera_id: manager
            for manager in active_physical_streams.values()
            if manager.camera_id in camera_ids
        }

    threshold = app.config.get('CONCEALMENT_CUE_THRESHOLD')
    camera_statuses = []
    for camera in cameras:
        manager = managers.get(camera.id)
        if manager is not None:
            camera_settings = manager.settings_cache
        elif settings is not None:
            camera_settings = {
                'face_recognition_enabled': settings.face_recognition_enabled,
                'unidentified_email_alerts_enabled': settings.unidentified_email_alerts_enabled,
                'recipient_email': settings.recipient_email,
            }
        else:
            camera_settings = {}
        camera_statuses.append({
            'camera_id': camera.id,
            'camera_name': camera.name,
            'camera_is_public': bool(camera.is_public),
            'is_active': bool(camera.is_active and camera.archived_at is None),
            'is_connected': bool(manager and manager.is_connected()),
            'face_recognition_enabled': bool(
                camera_settings.get('face_recognition_enabled', True)
            ),
            'cue_scan_enabled': threshold is not None,
            'email': {
                'preference_enabled': bool(
                    camera_settings.get('unidentified_email_alerts_enabled', False)
                ),
                'recipient_configured': bool(camera_settings.get('recipient_email')),
                'smtp_username_configured': bool(app.config.get('SMTP_USERNAME')),
                'smtp_password_configured': bool(app.config.get('SMTP_PASSWORD')),
            },
            'scan': concealment_scan_service.camera_status(camera.id),
        })

    service_status = concealment_scan_service.status()
    if settings is not None and settings.last_unidentified_email_at is not None:
        cooldown_minutes = max(1, settings.critical_email_cooldown_minutes or 15)
        elapsed_seconds = max(
            0,
            (datetime.utcnow() - settings.last_unidentified_email_at).total_seconds(),
        )
        cooldown_remaining_seconds = max(0, cooldown_minutes * 60 - elapsed_seconds)
    else:
        cooldown_minutes = max(
            1,
            (settings.critical_email_cooldown_minutes if settings else 15) or 15,
        )
        cooldown_remaining_seconds = 0

    for camera_status in camera_statuses:
        camera_status['email']['cooldown_minutes'] = cooldown_minutes
        camera_status['email']['cooldown_active'] = cooldown_remaining_seconds > 0
        camera_status['email']['cooldown_remaining_seconds'] = int(
            cooldown_remaining_seconds
        )

    return jsonify({
        'enabled': threshold is not None,
        'threshold': threshold,
        'model_loaded': service_status['model_loaded'],
        'worker_alive': service_status['worker_alive'],
        'cameras': camera_statuses,
    })


@app.route('/admin/system/save', methods=['POST'])
@admin_required
def admin_system_save():
    system_settings = get_system_settings()
    try:
        recorded_footage_retention_days = int(request.form.get(
            'retention_days',
            system_settings.recorded_footage_retention_days,
        ))
        request_video_retention_days = int(request.form.get(
            'request_video_retention_days',
            system_settings.request_video_retention_days,
        ))
    except (TypeError, ValueError):
        flash('Retention periods must be whole numbers between 1 and 365 days.', 'danger')
        return redirect(url_for('admin_system'))
    if not 1 <= recorded_footage_retention_days <= 365 or not 1 <= request_video_retention_days <= 365:
        flash('Retention periods must be between 1 and 365 days.', 'danger')
        return redirect(url_for('admin_system'))

    settings = Settings.query.filter_by(user_id=current_user.id).first()
    if not settings:
        settings = Settings(user_id=current_user.id)
        db.session.add(settings)

    try:
        object_detection_confidence = float(request.form.get(
            'object_detection_confidence',
            settings.object_detection_confidence,
        ))
        fire_confidence = parse_confidence_threshold(
            request.form.get('fire_confidence', object_detection_confidence),
            'Fire confidence threshold',
        )
        smoke_confidence = parse_confidence_threshold(
            request.form.get('smoke_confidence', object_detection_confidence),
            'Smoke confidence threshold',
        )
    except (TypeError, ValueError) as exc:
        flash(str(exc), 'danger')
        return redirect(url_for('admin_system'))

    system_settings.recorded_footage_retention_days = recorded_footage_retention_days
    system_settings.request_video_retention_days = request_video_retention_days

    settings.allow_registration = 'allow_registration' in request.form
    settings.require_disclaimer = 'require_disclaimer' in request.form
    settings.session_timeout_enabled = 'session_timeout_enabled' in request.form
    settings.session_timeout_minutes = int(request.form.get('session_timeout_minutes', 60))

    settings.yolo_enabled = 'yolo_enabled' in request.form
    settings.face_recognition_enabled = 'face_recognition_enabled' in request.form
    settings.object_detection_confidence = object_detection_confidence
    settings.face_recognition_confidence = float(request.form.get('face_recognition_confidence', 0.6))
    settings.active_classes = request.form.get('active_classes', 'fire,smoke')    
    settings.frame_process_interval = int(request.form.get('frame_process_interval', 3))

    detection_thresholds = get_or_create_detection_threshold_settings(
        current_user.id,
        object_detection_confidence,
    )
    detection_thresholds.fire_confidence = fire_confidence
    detection_thresholds.smoke_confidence = smoke_confidence

    db.session.commit()
    record_audit_event(current_user.id, 'settings', 'Updated system settings', 'Admin', get_client_ip())

    refresh_stream_settings(settings)
    _retention_cleanup_wakeup.set()

    flash('System settings updated.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_recordings', methods=['POST'])
@admin_required
def admin_clear_recordings():
    cleared_count = 0
    for user_dir in glob.glob(os.path.join(BASE_RECORDINGS_DIR, '*', 'recordings')):
        for filename in os.listdir(user_dir):
            if filename.endswith('.webm'):
                os.remove(os.path.join(user_dir, filename))
                cleared_count += 1
    record_audit_event(current_user.id, 'clear', f'Cleared {cleared_count} recording file(s)', 'Admin', get_client_ip())
    flash('All recordings were cleared.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_events', methods=['POST'])
@admin_required
def admin_system_clear_events():
    cleared_count = clear_admin_vision_events_and_snapshots()
    db.session.commit()
    record_audit_event(
        current_user.id, 'clear',
        f'Cleared {cleared_count} admin computer-vision event log(s); audit trail preserved',
        'Admin', get_client_ip(),
    )
    flash('Computer-vision event logs for all admin accounts were cleared. User events and audit logs were preserved.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/clear_faces', methods=['POST'])
@admin_required
def admin_clear_faces():
    cleared_count = 0
    for user_dir in glob.glob(os.path.join(BASE_RECORDINGS_DIR, '*', 'known_faces')):
        cleared_count += sum(len(files) for _, _, files in os.walk(user_dir))
        shutil.rmtree(user_dir, ignore_errors=True)
        os.makedirs(user_dir, exist_ok=True)
    record_audit_event(current_user.id, 'clear', f'Cleared {cleared_count} known face file(s)', 'Admin', get_client_ip())
    flash('All known faces were cleared.', 'success')
    return redirect(url_for('admin_system'))


@app.route('/admin/system/factory_reset', methods=['POST'])
@admin_required
def admin_factory_reset():
    cleared_count = clear_vision_events_and_snapshots()
    cameras_to_archive = Camera.query.filter(Camera.archived_at.is_(None)).all()
    Camera.query.filter(Camera.archived_at.is_(None)).update(
        {Camera.archived_at: datetime.utcnow()},
        synchronize_session=False,
    )
    Settings.query.delete()
    db.session.commit()
    for camera in cameras_to_archive:
        release_camera_streams(camera.id)
    record_audit_event(
        current_user.id, 'factory_reset',
        f'Factory reset completed; removed {cleared_count} computer-vision event log(s)',
        'Admin', get_client_ip(),
    )
    flash('Factory reset completed.', 'success')
    return redirect(url_for('admin_system'))

@app.route('/add_camera', methods=['POST'])
@login_required
def add_camera():
    source = request.form.get('source')
    name = request.form.get('name')
    zone = request.form.get('zone', 'Main Entrance').strip() or 'Main Entrance'
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
        zone=zone,
        is_public=request.form.get('is_public', 'false').lower() == 'true',
    )
    db.session.add(new_cam)
    db.session.commit()
    acquire_stream(current_user.id, new_cam)
    if is_admin_user(current_user):
        record_audit_event(
            current_user.id, 'add',
            f"Added camera \"{new_cam.name}\" (#{new_cam.id})",
            'Admin', get_client_ip(),
        )
    flash('Camera added.', 'success')

    return redirect(redirect_target)


@app.route('/admin/cameras/<int:camera_id>/toggle_public', methods=['POST'])
@admin_required
def admin_toggle_camera_public(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.archived_at is not None:
        flash('Archived cameras cannot be changed.', 'danger')
        return redirect(url_for('admin_cctv', include_archived=1))
    if camera.user_id != current_user.id and not camera.is_public:
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('admin_cctv'))

    camera.is_public = not camera.is_public
    db.session.commit()
    visibility = 'public' if camera.is_public else 'private'
    record_audit_event(
        current_user.id, 'settings',
        f"Set camera \"{camera.name}\" (#{camera.id}) to {visibility}",
        'Admin', get_client_ip(),
    )
    flash(f'Camera "{camera.name}" is now {visibility}.', 'success')
    return redirect(url_for('admin_cctv'))


@app.route('/delete_camera/<int:camera_id>', methods=['POST'])
@login_required
def delete_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    is_admin = is_admin_user(current_user)
    
    if not is_admin and (camera.user_id != current_user.id or camera.is_public):
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('index'))
    
    if is_admin:
        redirect_target = url_for('admin_cctv')
    else:
        redirect_target = url_for('index')

    try:
        camera_name = camera.name
        camera.archived_at = datetime.utcnow()
        db.session.commit()
        release_camera_streams(camera.id)
        if is_admin:
            record_audit_event(
                current_user.id, 'remove',
                f"Removed camera \"{camera_name}\" (#{camera_id})",
                'Admin', get_client_ip(),
            )
        flash(f'Camera "{camera_name}" archived. Its request and event history was preserved.', 'success')
    except Exception as e:
        flash(f'Error deleting camera: {e}', 'danger')
        
    return redirect(redirect_target)


@app.route('/restore_camera/<int:camera_id>', methods=['POST'])
@login_required
def restore_camera(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.user_id != current_user.id and not is_admin_user(current_user):
        flash('Unauthorized action.', 'danger')
        return redirect(url_for('index'))

    camera.archived_at = None
    db.session.commit()
    if is_admin_user(current_user):
        record_audit_event(
            current_user.id, 'restore',
            f"Restored camera \"{camera.name}\" (#{camera.id})",
            'Admin', get_client_ip(),
        )
    flash(f'Camera "{camera.name}" restored.', 'success')
    return redirect(url_for('admin_cctv', include_archived=1) if is_admin_user(current_user) else url_for('index'))

@app.route('/video_feed/<int:camera_id>')
@login_required
def video_feed(camera_id):
    camera = Camera.query.get_or_404(camera_id)
    if camera.archived_at is not None:
        return "Camera is archived", 404
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
        cameras = Camera.query.filter(Camera.archived_at.is_(None)).order_by(Camera.id).all()
    else:
        cameras = Camera.query.filter(
            Camera.archived_at.is_(None),
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
            stream_owner_id = camera.user_id if camera.is_public else user_id
            with app.app_context():
                user_settings = Settings.query.filter_by(user_id=stream_owner_id).first()
            manager = VideoStreamManager(
                stream_owner_id,
                camera.id,
                camera.source,
                user_settings,
                camera.name,
                getattr(camera, 'zone', 'Main Entrance'),
                getattr(camera, 'is_public', False),
            )
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


def release_camera_streams(camera_id):
    with stream_lock:
        subscribers = [
            (user_id, camera_streams[camera_id])
            for user_id, camera_streams in active_user_streams.items()
            if camera_id in camera_streams
        ]
    for user_id, manager in subscribers:
        release_stream(user_id, camera_id, manager.scope_key, manager)


def refresh_stream_settings(settings, user_id=None):
    """Apply settings to active workers, optionally limited to one owner."""
    with stream_lock:
        managers = {
            manager for manager in active_physical_streams.values()
            if user_id is None or manager.user_id == user_id
        }
        for manager in managers:
            manager.update_settings(settings)


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
    
    user_faces_dir = user_faces_directory(current_user.id)
    known_faces_list = []
    known_face_details = []
    if os.path.exists(user_faces_dir):
        known_faces_list = sorted(
            name for name in os.listdir(user_faces_dir)
            if os.path.isdir(os.path.join(user_faces_dir, name))
        )
        known_face_details = [
            {
                'name': name,
                'images': sorted(
                    filename for filename in os.listdir(os.path.join(user_faces_dir, name))
                    if allowed_file(filename)
                    and os.path.isfile(os.path.join(user_faces_dir, name, filename))
                ),
            }
            for name in known_faces_list
        ]

    if request.method == 'POST':
        user_settings.yolo_enabled = 'yolo_enabled' in request.form
        user_settings.face_recognition_enabled = 'face_recognition_enabled' in request.form
        user_settings.email_alerts_enabled = 'email_alerts_enabled' in request.form
        user_settings.unidentified_email_alerts_enabled = (
            'unidentified_email_alerts_enabled' in request.form
        )
        recipient_email = request.form.get('recipient_email', '').strip().lower()
        if not recipient_email or '@' not in recipient_email:
            flash('Enter a valid alert email address.', 'danger')
            return redirect(url_for('settings'))
        user_settings.recipient_email = recipient_email[:150]
        try:
            object_detection_confidence = max(0.1, min(1.0, float(request.form.get('object_detection_confidence', 0.5))))
            fire_confidence = parse_confidence_threshold(
                request.form.get('fire_confidence', object_detection_confidence),
                'Fire confidence threshold',
            )
            smoke_confidence = parse_confidence_threshold(
                request.form.get('smoke_confidence', object_detection_confidence),
                'Smoke confidence threshold',
            )
            user_settings.frame_process_interval = max(1, min(30, int(request.form.get('frame_process_interval', 3))))
            user_settings.object_detection_confidence = object_detection_confidence
            user_settings.face_recognition_confidence = max(0.1, min(1.0, float(request.form.get('face_recognition_confidence', 0.6))))
            user_settings.critical_email_cooldown_minutes = max(1, min(1440, int(request.form.get('critical_email_cooldown_minutes', 15))))
        except (TypeError, ValueError) as exc:
            flash(f'Detection settings must contain valid values: {exc}', 'danger')
            return redirect(url_for('settings'))
        detection_thresholds = get_or_create_detection_threshold_settings(
            current_user.id,
            object_detection_confidence,
        )
        detection_thresholds.fire_confidence = fire_confidence
        detection_thresholds.smoke_confidence = smoke_confidence
        db.session.commit()
        flash("Settings Updated", "success")
        
        refresh_stream_settings(user_settings, current_user.id)
        
        return redirect(url_for('settings'))
        
    fire_confidence, smoke_confidence = get_detection_threshold_values(
        current_user.id,
        user_settings.object_detection_confidence if user_settings else 0.5,
    )
    return render_template(
        'user_system.html',
        settings=user_settings,
        fire_confidence=fire_confidence,
        smoke_confidence=smoke_confidence,
        known_faces=known_faces_list,
        known_face_details=known_face_details,
    )


@app.route('/update_profile', methods=['POST'])
@login_required
def update_profile():
    username = request.form.get('username', '').strip()
    email = request.form.get('email', '').strip().lower()

    if not username or not email:
        flash('Username and email are required.', 'danger')
        return redirect(url_for('settings'))

    email_format = re.compile(
        r"[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
        r"@(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,63}"
    )
    if not email_format.fullmatch(email):
        flash('Enter a valid email address.', 'danger')
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
    return store_face_uploads(request.form.get('name'), request.files.getlist('file'))


@app.route('/add_face_photo/<name>', methods=['POST'])
@login_required
def add_face_photo(name):
    return store_face_uploads(name, request.files.getlist('file'))


def store_face_uploads(name, uploads):
    safe_name = safe_face_name(name)
    if not safe_name:
        flash('Enter a valid person name.', 'danger')
        return redirect(url_for('settings'))

    files = [upload for upload in uploads if upload and upload.filename]
    if not files:
        flash('Select at least one face photo to upload.', 'danger')
        return redirect(url_for('settings'))

    if any(not allowed_file(upload.filename) for upload in files):
        flash('Invalid file type. Allowed: png, jpg, jpeg.', 'danger')
        return redirect(url_for('settings'))

    save_dir = os.path.join(user_faces_directory(current_user.id), safe_name)
    saved_paths = []

    def remove_saved_files():
        for saved_path in saved_paths:
            try:
                os.remove(saved_path)
            except FileNotFoundError:
                continue
            except OSError:
                app.logger.exception('Unable to roll back uploaded face photo %s', saved_path)

    try:
        os.makedirs(save_dir, exist_ok=True)
        for upload in files:
            requested_filename = secure_filename(upload.filename)
            if not requested_filename or not allowed_file(requested_filename):
                raise ValueError('A selected filename is invalid.')

            stem, extension = os.path.splitext(requested_filename)
            filename = requested_filename
            suffix = 1
            while os.path.exists(os.path.join(save_dir, filename)):
                filename = f'{stem}-{suffix}{extension}'
                suffix += 1

            destination = os.path.join(save_dir, filename)
            upload.save(destination)
            saved_paths.append(destination)
    except ValueError as exc:
        remove_saved_files()
        flash(str(exc), 'danger')
        return redirect(url_for('settings'))
    except OSError:
        app.logger.exception('Unable to save uploaded face photos for user %s', current_user.id)
        remove_saved_files()
        flash('Unable to save the selected face photos.', 'danger')
        return redirect(url_for('settings'))

    invalidate_face_cache(current_user.id)
    flash(f'{len(saved_paths)} photo(s) added for "{safe_name}".', 'success')
    return redirect(url_for('settings'))

@app.route('/delete_face/<name>', methods=['POST'])
@login_required
def delete_face(name):
    safe_name = safe_face_name(name)
    if not safe_name:
        flash('Face not found.', 'danger')
        return redirect(url_for('settings'))

    face_dir = os.path.join(user_faces_directory(current_user.id), safe_name)
    
    if os.path.isdir(face_dir):
        try:
            shutil.rmtree(face_dir)
            invalidate_face_cache(current_user.id)
            flash(f'Face "{safe_name}" deleted.', 'success')
        except OSError:
            app.logger.exception('Unable to delete face directory for user %s', current_user.id)
            flash('Unable to delete this face profile.', 'danger')
    else:
        flash('Face not found.', 'danger')
        
    return redirect(url_for('settings'))


@app.route('/delete_face_image/<name>/<filename>', methods=['POST'])
@login_required
def delete_face_image(name, filename):
    safe_name = safe_face_name(name)
    safe_filename = secure_filename(filename)
    image_path = os.path.join(user_faces_directory(current_user.id), safe_name, safe_filename)

    if not safe_name or not safe_filename or not allowed_file(safe_filename) or not os.path.isfile(image_path):
        flash('Face photo not found.', 'danger')
        return redirect(url_for('settings'))

    try:
        os.remove(image_path)
        invalidate_face_cache(current_user.id)
        flash(f'Photo "{safe_filename}" removed from "{safe_name}".', 'success')
    except OSError:
        app.logger.exception('Unable to delete face photo for user %s', current_user.id)
        flash('Unable to delete this face photo.', 'danger')

    return redirect(url_for('settings'))


@app.route('/rename_face/<name>', methods=['POST'])
@login_required
def rename_face(name):
    old_name = safe_face_name(name)
    new_name = safe_face_name(request.form.get('new_name'))
    if not old_name or not new_name:
        flash('Enter a valid person name.', 'danger')
        return redirect(url_for('settings'))

    old_dir = os.path.join(user_faces_directory(current_user.id), old_name)
    new_dir = os.path.join(user_faces_directory(current_user.id), new_name)
    if not os.path.isdir(old_dir):
        flash('Face not found.', 'danger')
        return redirect(url_for('settings'))
    if old_name == new_name:
        flash('The person name is unchanged.', 'success')
        return redirect(url_for('settings'))
    if os.path.exists(new_dir):
        flash('A person with that name already exists.', 'danger')
        return redirect(url_for('settings'))

    try:
        os.rename(old_dir, new_dir)
        invalidate_face_cache(current_user.id)
        flash(f'Face profile renamed to "{new_name}".', 'success')
    except OSError:
        app.logger.exception('Unable to rename face profile for user %s', current_user.id)
        flash('Unable to rename this face profile.', 'danger')

    return redirect(url_for('settings'))


@app.route('/known_face_image/<name>/<filename>')
@login_required
def face_image(name, filename):
    safe_name = safe_face_name(name)
    safe_filename = secure_filename(filename)
    if not safe_name or not safe_filename or not allowed_file(safe_filename):
        return '', 404

    face_dir = os.path.join(user_faces_directory(current_user.id), safe_name)
    return send_from_directory(face_dir, safe_filename)


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
    cameras = Camera.query.filter(
        or_(Camera.user_id == current_user.id, Camera.is_public.is_(True))
    ).order_by(Camera.name, Camera.id).all()
    events = get_user_vision_events(current_user.id)
    event_items = [
        {
            'id': event.id,
            'timestamp': (
                to_manila_time(event.timestamp).strftime('%H:%M:%S PHT')
                if event.timestamp else '—'
            ),
            'date': (
                to_manila_time(event.timestamp).strftime('%b %d, %Y')
                if event.timestamp else ''
            ),
            'type': event.event_type or 'NOTIFICATION',
            'event_type': event.event_type,
            'camera': event.camera.name if event.camera else event.source_name or 'System',
            'source_name': event.source_name,
            'description': event.description,
            'snapshot_url': url_for('event_snapshot', event_id=event.id) if event.snapshot_path else None,
        }
        for event in events
    ]
    return render_template('user_event.html', cameras=cameras, events=event_items)

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
    events = get_user_vision_events(current_user.id)
    result = []
    for event in events:
        result.append({
            'id': event.id,
            'timestamp': to_manila_time(event.timestamp).isoformat(),
            'camera': event.camera.name if event.camera else event.source_name or 'System',
            'source_name': event.source_name,
            'event_type': event.event_type,
            'description': event.description,
            'snapshot_url': url_for('event_snapshot', event_id=event.id) if event.snapshot_path else None,
        })
    return jsonify(result)


def get_user_vision_events(user_id):
    return DetectionEvent.query.filter_by(
        user_id=user_id,
        event_category='vision',
    ).order_by(DetectionEvent.timestamp.desc()).all()

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
    event = DetectionEvent.query.filter_by(
        id=event_id,
        user_id=current_user.id,
    ).first()
    if event is None:
        return jsonify({"error": "Event not found"}), 404
    
    try:
        snapshot_path = get_event_snapshot_path(event)
        db.session.delete(event)
        db.session.commit()
        remove_event_snapshot_file(snapshot_path)
        return jsonify({"success": True, "message": "Event deleted"})
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


def clear_vision_events_and_snapshots(user_id=None):
    events_query = DetectionEvent.query
    if user_id is not None:
        events_query = events_query.filter_by(user_id=user_id)
    return _delete_detection_events_and_snapshots(events_query)


def clear_admin_vision_events_and_snapshots():
    admin_user_ids = db.session.query(User.id).filter(
        func.lower(func.trim(User.role)) == 'admin',
    )
    events_query = DetectionEvent.query.filter(
        DetectionEvent.user_id.in_(admin_user_ids),
        DetectionEvent.event_category == 'vision',
    )
    return _delete_detection_events_and_snapshots(events_query)


def _delete_detection_events_and_snapshots(events_query):
    events = events_query.all()
    snapshot_paths = [get_event_snapshot_path(event) for event in events]
    cleared_count = events_query.delete(synchronize_session=False)
    db.session.commit()
    for snapshot_path in snapshot_paths:
        remove_event_snapshot_file(snapshot_path)
    return cleared_count


def get_event_snapshot_path(event):
    if (
        not event.snapshot_path
        or not re.fullmatch(r'\d+\.jpg', event.snapshot_path, re.IGNORECASE)
        or not str(event.user_id).isdecimal()
    ):
        return None
    return os.path.join(
        BASE_RECORDINGS_DIR,
        str(event.user_id),
        'event_snapshots',
        event.snapshot_path,
    )


def remove_event_snapshot_file(snapshot_path):
    if not snapshot_path:
        return
    try:
        os.remove(snapshot_path)
    except FileNotFoundError:
        pass
    except OSError:
        app.logger.exception('Unable to remove event snapshot %s', snapshot_path)


@app.route('/events/<int:event_id>/snapshot')
@login_required
def event_snapshot(event_id):
    event = DetectionEvent.query.filter_by(id=event_id, user_id=current_user.id).first_or_404()
    snapshot_path = get_event_snapshot_path(event)
    if not snapshot_path or not os.path.isfile(snapshot_path):
        return 'Snapshot is not available.', 404
    try:
        with open(snapshot_path, 'rb') as snapshot_file:
            snapshot_bytes = snapshot_file.read()
    except OSError:
        app.logger.exception('Unable to read event snapshot %s', snapshot_path)
        return 'Snapshot is not available.', 404
    return send_file(BytesIO(snapshot_bytes), mimetype='image/jpeg', download_name=f'event-{event_id}.jpg')


@app.route('/admin/events/<int:event_id>/snapshot')
@admin_required
def admin_event_snapshot(event_id):
    event = DetectionEvent.query.get_or_404(event_id)
    snapshot_path = get_event_snapshot_path(event)
    if not snapshot_path or not os.path.isfile(snapshot_path):
        return 'Snapshot is not available.', 404
    try:
        with open(snapshot_path, 'rb') as snapshot_file:
            snapshot_bytes = snapshot_file.read()
    except OSError:
        app.logger.exception('Unable to read event snapshot %s', snapshot_path)
        return 'Snapshot is not available.', 404
    return send_file(BytesIO(snapshot_bytes), mimetype='image/jpeg', download_name=f'event-{event_id}.jpg')


@app.route('/api/clear_private_events', methods=['POST'])
@login_required
def api_clear_private_events():
    try:
        private_camera_ids = db.session.query(Camera.id).filter(
            Camera.user_id == current_user.id,
            Camera.is_public.is_(False),
        )
        private_events = DetectionEvent.query.filter(
            DetectionEvent.user_id == current_user.id,
            DetectionEvent.camera_id.in_(private_camera_ids),
        )
        private_event_rows = private_events.all()
        cleared_event_ids = [event.id for event in private_event_rows]
        snapshot_paths = [get_event_snapshot_path(event) for event in private_event_rows]
        cleared_count = private_events.delete(synchronize_session=False)
        db.session.commit()
        for snapshot_path in snapshot_paths:
            remove_event_snapshot_file(snapshot_path)
        return jsonify({
            "success": True,
            "message": "Private camera events cleared",
            "cleared_count": cleared_count,
            "cleared_event_ids": cleared_event_ids,
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

# --- INITIALIZATION ---
def init_app():
    global yolo_model, yolo_model_object, behavior_pose_model
    global detection_pool, ALL_YOLO_CLASS_NAMES, _retention_cleanup_thread

    with app.app_context():
        ensure_database_schema()
        get_system_settings()
        cleanup_expired_recordings()

    if _retention_cleanup_thread is None or not _retention_cleanup_thread.is_alive():
        _retention_cleanup_wakeup.clear()
        _retention_cleanup_thread = threading.Thread(target=retention_cleanup_worker, daemon=True)
        _retention_cleanup_thread.start()

    if yolo_model is None:
        yolo_model = YOLO(MODEL_PATH)
        if yolo_model.names:
            ALL_YOLO_CLASS_NAMES = list(yolo_model.names.values())

    if yolo_model_object is None:
        yolo_model_object = YOLO(MODEL_PATH_OBJECT)

    if behavior_pose_model is None and BEHAVIOR_POSE_MODEL_PATH:
        pose_model_path = os.path.abspath(os.path.join(APP_ROOT, BEHAVIOR_POSE_MODEL_PATH))
        if not os.path.isfile(pose_model_path):
            raise FileNotFoundError(
                f'BEHAVIOR_POSE_MODEL does not point to an existing file: {pose_model_path}'
            )
        behavior_pose_model = YOLO(pose_model_path)
        app.logger.info('Loaded optional behavior pose model from %s', pose_model_path)

    if detection_pool is None:
        detection_pool = ThreadPoolExecutor(max_workers=MAX_DETECTION_POOL_WORKERS)

    with app.app_context():
        for camera in Camera.query.filter_by(is_active=True, archived_at=None).all():
            acquire_stream(camera.user_id, camera)

    print("App Initialized.")


def main():
    parser = argparse.ArgumentParser(description='Run the smart security app')
    parser.add_argument('--create-admin', action='store_true', help='Create an admin account from the command line')
    parser.add_argument('--username', help='Username for the admin account')
    parser.add_argument('--email', help='Email address for the admin account')
    parser.add_argument('--password', help='Password for the admin account')
    parser.add_argument(
        '--legacy-otp',
        action='store_true',
        help='Use authenticator app TOTP instead of email OTP (legacy/testing mode)',
    )
    args = parser.parse_args()
    app.config['LEGACY_OTP'] = args.legacy_otp

    if args.create_admin:
        try:
            admin_user = create_admin_user(args.username, args.email, args.password)
            print(f"Admin created successfully: {admin_user.email}")
            return 0
        except Exception as exc:
            print(f"Admin creation failed: {exc}")
            return 1

    if args.legacy_otp:
        print("[OTP MODE] Legacy mode active — login verification uses authenticator app (TOTP).")
    else:
        smtp_user = app.config.get('SMTP_USERNAME', '')
        smtp_pass = app.config.get('SMTP_PASSWORD', '')
        if not smtp_user or not smtp_pass:
            print(
                "[WARNING] Email OTP mode is active but SMTP is not configured.\n"
                "          Users will NOT receive login verification codes by email.\n"
                "          Set SMTP_USERNAME and SMTP_PASSWORD in your .env file,\n"
                "          or start with --legacy-otp to use the authenticator app instead."
            )
        else:
            print("[OTP MODE] Email OTP mode active — login verification codes will be sent by email.")

    init_app()
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '5000')), threaded=True, debug=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
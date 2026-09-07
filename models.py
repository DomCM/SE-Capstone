from datetime import datetime

from flask_login import UserMixin
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='user')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime, nullable=True)
    totp_secret = db.Column(db.String(500), nullable=True)
    totp_enabled = db.Column(db.Boolean, nullable=False, default=False)

    @property
    def banned(self):
        return self.role == 'banned'

    cameras = db.relationship('Camera', backref='owner', lazy=True)
    settings = db.relationship('Settings', backref='owner', uselist=False, lazy=True)


class Camera(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), default='My Camera')
    source = db.Column(db.String(500), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    is_public = db.Column(db.Boolean, default=False)


class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    yolo_enabled = db.Column(db.Boolean, default=True)
    yolo_object_enabled = db.Column(db.Boolean, default=True)
    face_recognition_enabled = db.Column(db.Boolean, default=True)
    confidence_threshold = db.Column(db.Float, default=0.4)
    object_detection_confidence = db.Column(db.Float, default=0.5)
    face_recognition_confidence = db.Column(db.Float, default=0.6)
    active_classes = db.Column(db.String(500), default='fire,smoke')
    allow_registration = db.Column(db.Boolean, default=True)
    require_disclaimer = db.Column(db.Boolean, default=False)
    session_timeout_enabled = db.Column(db.Boolean, default=False)
    session_timeout_minutes = db.Column(db.Integer, default=60)
    email_alerts_enabled = db.Column(db.Boolean, default=False)
    recipient_email = db.Column(db.String(150))
    scale_down_amount = db.Column(db.Integer, default=2)
    frame_process_interval = db.Column(db.Integer, default=3)


class OtpChallenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    purpose = db.Column(db.String(30), nullable=False)
    code_hash = db.Column(db.String(256), nullable=False)
    expires_at = db.Column(db.DateTime, nullable=False)
    attempts = db.Column(db.Integer, nullable=False, default=0)
    used_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref='otp_challenges', lazy=True)


class EventLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.now)
    source_name = db.Column(db.String(100))
    event_type = db.Column(db.String(50))
    description = db.Column(db.String(500))
    confidence = db.Column(db.Float, nullable=True)
    ip_address = db.Column(db.String(45), nullable=True)
    user = db.relationship('User', backref='event_logs', lazy=True)
    camera = db.relationship('Camera', backref='event_logs', lazy=True)

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
            'login': 'Login', 'logout': 'Logout', 'create': 'Create',
            'delete': 'Delete', 'settings': 'Settings', 'motion': 'Motion',
            'alert': 'Alert', 'recording': 'Recording',
        }
        return mapping.get(self.action_type, self.action_type.title())

    @property
    def detail(self):
        return self.description or self.source_name or '-'


class RecordingRequest(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    camera_id = db.Column(db.Integer, db.ForeignKey('camera.id'), nullable=False)
    date_needed = db.Column(db.String(10), nullable=False)
    time_range = db.Column(db.String(100), nullable=False)
    reason = db.Column(db.String(1000), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='pending')
    rejection_reason = db.Column(db.String(1000), nullable=True)
    video_filename = db.Column(db.String(255), nullable=True)
    video_path = db.Column(db.String(500), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    fulfilled_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship('User', backref='recording_requests')
    camera = db.relationship('Camera')

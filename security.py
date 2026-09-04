import base64
import hashlib
import os
import secrets
import smtplib
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import pyotp
from cryptography.fernet import Fernet, InvalidToken
from flask import request
from sqlalchemy import inspect, text, or_
from werkzeug.security import generate_password_hash


app = None
db = None
User = Camera = Settings = OtpChallenge = EventLog = RecordingRequest = None
BASE_RECORDINGS_DIR = 'users_data'
AUDIT_EVENT_TYPES = {'login', 'logout', 'create', 'delete', 'settings'}


def configure(application, database, models, recordings_dir='users_data'):
    global app, db, User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest, BASE_RECORDINGS_DIR
    app = application
    db = database
    User, Camera, Settings, OtpChallenge, EventLog, RecordingRequest = models
    BASE_RECORDINGS_DIR = recordings_dir


def get_user_role(user):
    if user is None:
        return 'user'
    return str(getattr(user, 'role', None) or 'user').strip().lower() or 'user'


def is_admin_user(user_or_email):
    if user_or_email is None:
        return False
    admin_emails = [email.strip().lower() for email in os.environ.get('ADMIN_EMAILS', '').split(',') if email.strip()]
    if isinstance(user_or_email, str):
        email = user_or_email.strip().lower()
        if email in admin_emails:
            return True
        user = User.query.filter_by(email=email).first()
        return bool(user and get_user_role(user) == 'admin')
    email = getattr(user_or_email, 'email', '').strip().lower()
    return email in admin_emails or get_user_role(user_or_email) == 'admin'


def get_client_ip():
    forwarded_for = request.headers.get('X-Forwarded-For') if request else None
    if forwarded_for:
        return forwarded_for.split(',')[0].strip() or request.remote_addr
    return request.remote_addr if request else None


def record_audit_event(user_id, event_type, description, source_name='System', ip_address=None):
    if user_id is None:
        return None
    normalized_type = (event_type or 'settings').strip().lower()
    if normalized_type not in AUDIT_EVENT_TYPES:
        normalized_type = 'settings'
    safe_description = (description or '').strip() or f'{normalized_type.title()} action recorded'
    try:
        with app.app_context():
            entry = EventLog(user_id=user_id, source_name=source_name or 'System', event_type=normalized_type,
                             description=safe_description[:500], ip_address=ip_address or get_client_ip())
            db.session.add(entry)
            db.session.commit()
            return entry
    except Exception:
        db.session.rollback()
        return None


def get_dashboard_target(user):
    return 'admin_dashboard' if is_admin_user(user) else 'index'


def _security_fernet():
    secret = app.config['SECRET_KEY']
    if isinstance(secret, str):
        secret = secret.encode()
    return Fernet(base64.urlsafe_b64encode(hashlib.sha256(secret).digest()))


def encrypt_totp_secret(secret):
    return _security_fernet().encrypt(secret.encode()).decode()


def decrypt_totp_secret(encrypted_secret):
    if not encrypted_secret:
        return None
    try:
        return _security_fernet().decrypt(encrypted_secret.encode()).decode()
    except (InvalidToken, ValueError):
        return None


def get_or_create_totp_secret(user):
    secret = decrypt_totp_secret(user.totp_secret)
    if secret:
        return secret
    secret = pyotp.random_base32()
    user.totp_secret = encrypt_totp_secret(secret)
    db.session.commit()
    return secret


def send_security_email(user, subject, body):
    sender = app.config['SMTP_USERNAME']
    password = app.config['SMTP_PASSWORD']
    if not sender or not password:
        return False
    try:
        message = MIMEText(body, 'plain')
        message['From'], message['To'], message['Subject'] = sender, user.email, subject
        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)
        return True
    except Exception as exc:
        app.logger.warning('Security email failed: %s', exc)
        return False


def create_email_otp(user):
    OtpChallenge.query.filter_by(user_id=user.id, purpose='password_reset', used_at=None).update({'used_at': datetime.utcnow()})
    code = f'{secrets.randbelow(1000000):06d}'
    db.session.add(OtpChallenge(user_id=user.id, purpose='password_reset', code_hash=generate_password_hash(code),
                                expires_at=datetime.utcnow() + timedelta(minutes=10)))
    db.session.commit()
    return code


def clear_audit_events_for_user(user_id=None):
    query = EventLog.query
    if user_id is not None:
        query = query.filter_by(user_id=user_id)
    return query.filter(EventLog.event_type.in_(list(AUDIT_EVENT_TYPES))).delete(synchronize_session=False)


def ensure_database_schema():
    with app.app_context():
        db.create_all()
        inspector = inspect(db.engine)
        user_columns = {column['name'] for column in inspector.get_columns('user')}
        for name, definition in [('role', "VARCHAR(20) NOT NULL DEFAULT 'user'"), ('totp_secret', 'VARCHAR(500)'), ('totp_enabled', 'BOOLEAN NOT NULL DEFAULT 0')]:
            if name not in user_columns:
                db.session.execute(text(f'ALTER TABLE user ADD COLUMN {name} {definition}'))
                db.session.commit()
        camera_columns = {column['name'] for column in inspector.get_columns('camera')}
        if 'is_public' not in camera_columns:
            db.session.execute(text('ALTER TABLE camera ADD COLUMN is_public BOOLEAN NOT NULL DEFAULT 0'))
            db.session.commit()


def create_admin_user(username, email, password):
    if not username or not email or not password:
        raise ValueError('Username, email, and password are required.')
    with app.app_context():
        ensure_database_schema()
        if User.query.filter((User.email == email) | (User.username == username)).first():
            raise ValueError('An account with that username or email already exists.')
        new_user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            role='admin',
            totp_secret=encrypt_totp_secret(pyotp.random_base32()),
            totp_enabled=False,
        )
        db.session.add(new_user)
        db.session.commit()
        db.session.add(Settings(user_id=new_user.id, recipient_email=email))
        db.session.commit()
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'known_faces'), exist_ok=True)
        os.makedirs(os.path.join(BASE_RECORDINGS_DIR, str(new_user.id), 'recordings'), exist_ok=True)
        return new_user

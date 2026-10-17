import os
import sys
import unittest

from sqlalchemy import inspect, text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import main


class EventAuditSeparationTests(unittest.TestCase):
    def setUp(self):
        self.app = main.app
        self.app.config.update(
            TESTING=True,
            SECRET_KEY='event-audit-test-secret',
            LEGACY_OTP=False,
            SQLALCHEMY_DATABASE_URI='sqlite:///:memory:',
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        with self.app.app_context():
            if 'event_log' in inspect(main.db.engine).get_table_names():
                main.db.session.execute(text('DROP TABLE event_log'))
                main.db.session.commit()
            main.db.drop_all()
            main.db.create_all()
            self.user = main.User(
                username='event-user',
                email='event-user@example.com',
                password='secret',
                role='user',
            )
            main.db.session.add(self.user)
            main.db.session.flush()
            self.admin = main.User(
                username='event-admin',
                email='event-admin@example.com',
                password='secret',
                role='admin',
            )
            main.db.session.add(self.admin)
            main.db.session.flush()
            self.camera = main.Camera(
                user_id=self.user.id,
                name='Front Gate',
                source='rtsp://example/camera',
            )
            main.db.session.add(self.camera)
            main.db.session.commit()
            self.user_id = self.user.id
            self.admin_id = self.admin.id
            self.camera_id = self.camera.id
        self.client = self.app.test_client()

    def login_as(self, user_id):
        with self.client.session_transaction() as session:
            session['privacy_agreed'] = True
            session['_user_id'] = str(user_id)
            session['_fresh'] = True

    def test_detection_and_audit_ids_are_independent(self):
        with self.app.app_context():
            main.security.record_audit_event(
                self.user_id,
                'login',
                'User logged in',
                'Auth',
            )
            detection = main.DetectionEvent(
                user_id=self.user_id,
                camera_id=self.camera_id,
                event_type='PERSON DETECTED',
                event_category='vision',
                severity='medium',
            )
            main.db.session.add(detection)
            main.db.session.commit()

            self.assertEqual(main.AuditLog.query.one().id, 1)
            self.assertEqual(main.DetectionEvent.query.one().id, 1)
            self.assertEqual(main.AuditLog.query.count(), 1)
            self.assertEqual(main.DetectionEvent.query.count(), 1)

    def test_clear_detection_events_preserves_audit_logs(self):
        with self.app.app_context():
            main.db.session.add_all([
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    event_type='PERSON DETECTED',
                    event_category='vision',
                    severity='medium',
                ),
                main.AuditLog(
                    user_id=self.user_id,
                    event_type='login',
                    description='User logged in',
                ),
            ])
            main.db.session.commit()

            self.assertEqual(main.clear_vision_events_and_snapshots(), 1)
            self.assertEqual(main.DetectionEvent.query.count(), 0)
            self.assertEqual(main.AuditLog.query.count(), 1)

    def test_user_event_api_and_delete_only_use_detection_events(self):
        with self.app.app_context():
            main.security.record_audit_event(
                self.user_id,
                'login',
                'User logged in',
                'Auth',
            )
            event = main.DetectionEvent(
                user_id=self.user_id,
                camera_id=self.camera_id,
                event_type='PERSON DETECTED',
                event_category='vision',
                severity='medium',
            )
            main.db.session.add(event)
            main.db.session.commit()
            detection_id = event.id

        self.login_as(self.user_id)
        response = self.client.get('/api/events')
        self.assertEqual(response.status_code, 200)
        self.assertEqual([entry['id'] for entry in response.json], [detection_id])

        response = self.client.post(f'/api/delete_event/{detection_id}')
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            self.assertEqual(main.DetectionEvent.query.count(), 0)
            self.assertEqual(main.AuditLog.query.count(), 1)

    def test_admin_log_and_report_routes_use_separate_models(self):
        with self.app.app_context():
            main.security.record_audit_event(
                self.admin_id,
                'login',
                'Audit only',
                'Auth',
            )
            main.db.session.add(main.DetectionEvent(
                user_id=self.user_id,
                camera_id=self.camera_id,
                event_type='PERSON DETECTED',
                description='Detection only',
                event_category='vision',
                severity='high',
            ))
            main.db.session.commit()

        self.login_as(self.admin_id)
        response = self.client.get('/admin/logs')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Audit only', response.data)
        self.assertNotIn(b'Detection only', response.data)

        response = self.client.get('/admin/reports?type=incident&timeframe=7d')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(b'Audit only', response.data)
        with self.app.app_context():
            report = main.build_report(
                main.DetectionEvent,
                report_type='incident',
                timeframe='7d',
                camera_model=main.Camera,
            )
            self.assertEqual(report['total_triggers'], 0)
            self.assertEqual(report['incident_logs'], [])

    def test_admin_reports_include_only_public_camera_events(self):
        with self.app.app_context():
            admin_private_camera = main.Camera(
                user_id=self.admin_id,
                name='Admin Private Camera',
                source='rtsp://example/admin-private',
            )
            user_public_camera = main.Camera(
                user_id=self.user_id,
                name='User Public Camera',
                source='rtsp://example/user-public',
                is_public=True,
            )
            admin_public_camera = main.Camera(
                user_id=self.admin_id,
                name='Admin Public Camera',
                source='rtsp://example/admin-public',
                is_public=True,
            )
            main.db.session.add_all([admin_private_camera, user_public_camera, admin_public_camera])
            main.db.session.flush()
            main.db.session.add_all([
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    source_name='User Private Camera',
                    event_type='PRIVATE USER ALERT',
                    event_category='vision',
                    severity='high',
                ),
                main.DetectionEvent(
                    user_id=self.admin_id,
                    camera_id=admin_private_camera.id,
                    source_name=admin_private_camera.name,
                    event_type='PRIVATE ADMIN ALERT',
                    event_category='vision',
                    severity='high',
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=user_public_camera.id,
                    source_name=user_public_camera.name,
                    event_type='PUBLIC USER ALERT',
                    event_category='vision',
                    severity='high',
                ),
                main.DetectionEvent(
                    user_id=self.admin_id,
                    camera_id=admin_public_camera.id,
                    source_name=admin_public_camera.name,
                    event_type='PUBLIC ADMIN ALERT',
                    event_category='vision',
                    severity='high',
                ),
            ])
            main.db.session.commit()

        self.login_as(self.admin_id)
        report_response = self.client.get('/admin/reports?timeframe=24h')
        self.assertEqual(report_response.status_code, 200)
        self.assertIn(b'PUBLIC USER ALERT', report_response.data)
        self.assertIn(b'PUBLIC ADMIN ALERT', report_response.data)
        self.assertNotIn(b'PRIVATE USER ALERT', report_response.data)
        self.assertNotIn(b'PRIVATE ADMIN ALERT', report_response.data)
        self.assertNotIn(b'Admin Private Camera', report_response.data)

        csv_response = self.client.get('/admin/reports/export?timeframe=24h')
        self.assertEqual(csv_response.status_code, 200)
        self.assertIn(b'PUBLIC USER ALERT', csv_response.data)
        self.assertIn(b'PUBLIC ADMIN ALERT', csv_response.data)
        self.assertNotIn(b'PRIVATE USER ALERT', csv_response.data)
        self.assertNotIn(b'PRIVATE ADMIN ALERT', csv_response.data)

    def test_legacy_combined_event_table_is_split_on_schema_initialization(self):
        with self.app.app_context():
            main.db.session.execute(text('''
                CREATE TABLE event_log (
                    id INTEGER PRIMARY KEY,
                    user_id INTEGER,
                    camera_id INTEGER,
                    timestamp DATETIME,
                    source_name VARCHAR(100),
                    event_type VARCHAR(50),
                    description VARCHAR(500),
                    confidence FLOAT,
                    event_category VARCHAR(30),
                    detector VARCHAR(30),
                    severity VARCHAR(30),
                    event_metadata TEXT,
                    snapshot_path VARCHAR(255),
                    ip_address VARCHAR(45),
                    actor_name VARCHAR(150),
                    actor_email VARCHAR(150)
                )
            '''))
            main.db.session.execute(text('''
                INSERT INTO event_log
                    (id, user_id, camera_id, event_type, description,
                     event_category, severity, snapshot_path)
                VALUES
                    (5, :user_id, NULL, 'login', 'User logged in',
                     'audit', NULL, NULL),
                    (6, :user_id, :camera_id, 'PERSON DETECTED', 'Person detected',
                     'vision', 'medium', '6.jpg')
            '''), {'user_id': self.user_id, 'camera_id': self.camera_id})
            main.db.session.commit()

            main.security.ensure_database_schema()

            audit = main.AuditLog.query.one()
            detection = main.DetectionEvent.query.one()
            table_names = inspect(main.db.engine).get_table_names()
            self.assertEqual(audit.id, 1)
            self.assertEqual(detection.id, 1)
            self.assertEqual(detection.snapshot_path, '6.jpg')
            self.assertNotIn('event_log', table_names)

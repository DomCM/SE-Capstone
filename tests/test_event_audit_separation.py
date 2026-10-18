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

    def test_admin_cctv_shows_and_allows_admins_to_manage_public_cameras(self):
        with self.app.app_context():
            other_admin = main.User(
                username='event-admin-b',
                email='event-admin-b@example.com',
                password='secret',
                role='admin',
            )
            main.db.session.add(other_admin)
            main.db.session.flush()
            public_camera = main.Camera(
                user_id=self.admin_id,
                name='Admin A Public Camera',
                source='rtsp://example/admin-a-public',
                is_public=True,
            )
            private_camera = main.Camera(
                user_id=self.admin_id,
                name='Admin A Private Camera',
                source='rtsp://example/admin-a-private',
            )
            main.db.session.add_all([public_camera, private_camera])
            main.db.session.commit()
            other_admin_id = other_admin.id
            public_camera_id = public_camera.id
            private_camera_id = private_camera.id

        self.login_as(other_admin_id)
        response = self.client.get('/admin/cctv')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Admin A Public Camera', response.data)
        self.assertNotIn(b'Admin A Private Camera', response.data)

        response = self.client.post(
            f'/admin/cameras/{public_camera_id}/toggle_public',
            follow_redirects=True,
        )
        self.assertEqual(response.status_code, 200)
        with self.app.app_context():
            self.assertFalse(main.db.session.get(main.Camera, public_camera_id).is_public)

        self.client.post(f'/admin/cameras/{private_camera_id}/toggle_public')
        with self.app.app_context():
            self.assertFalse(main.db.session.get(main.Camera, private_camera_id).is_public)

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

    def test_user_event_timestamps_are_rendered_in_manila_time(self):
        with self.app.app_context():
            event = main.DetectionEvent(
                user_id=self.user_id,
                camera_id=self.camera_id,
                event_type='PERSON DETECTED',
                event_category='vision',
                severity='medium',
                timestamp=main.datetime(2024, 1, 2, 12, 30),
            )
            main.db.session.add(event)
            main.db.session.commit()

        self.login_as(self.user_id)
        response = self.client.get('/events')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'20:30:00 PHT', response.data)
        self.assertIn(b'Jan 02, 2024', response.data)

        response = self.client.get('/api/events')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json[0]['timestamp'],
            '2024-01-02T20:30:00+08:00',
        )

        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'2024-01-02 20:30:00 PHT', response.data)

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

    def test_user_can_save_independent_fire_and_smoke_thresholds(self):
        with self.app.app_context():
            main.db.session.add(main.Settings(
                user_id=self.user_id,
                recipient_email='event-user@example.com',
            ))
            main.db.session.commit()

        self.login_as(self.user_id)
        response = self.client.post('/settings', data={
            'recipient_email': 'alerts@example.com',
            'frame_process_interval': '3',
            'object_detection_confidence': '0.5',
            'face_recognition_confidence': '0.6',
            'fire_confidence': '0.8',
            'smoke_confidence': '0.35',
        })
        self.assertEqual(response.status_code, 302)

        with self.app.app_context():
            thresholds = main.DetectionThresholdSettings.query.filter_by(
                user_id=self.user_id,
            ).one()
            self.assertEqual(thresholds.fire_confidence, 0.8)
            self.assertEqual(thresholds.smoke_confidence, 0.35)
            self.assertEqual(main.get_detection_class_threshold(
                'fire', thresholds.fire_confidence, thresholds.smoke_confidence, 0.5,
            ), 0.8)
            self.assertEqual(main.get_detection_class_threshold(
                'Smoke', thresholds.fire_confidence, thresholds.smoke_confidence, 0.5,
            ), 0.35)

        settings_page = self.client.get('/settings')
        self.assertEqual(settings_page.status_code, 200)
        self.assertIn(b'name="fire_confidence"', settings_page.data)
        self.assertIn(b'name="smoke_confidence"', settings_page.data)

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

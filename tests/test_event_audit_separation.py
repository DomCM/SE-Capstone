import os
import sys
import unittest
from datetime import datetime
from unittest.mock import patch

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

    def test_admin_clear_only_removes_admin_vision_events(self):
        with self.app.app_context():
            other_admin = main.User(
                username='event-admin-b',
                email='event-admin-b@example.com',
                password='secret',
                role='admin',
            )
            main.db.session.add(other_admin)
            main.db.session.flush()
            user_event = main.DetectionEvent(
                user_id=self.user_id,
                event_type='USER VISION EVENT',
                event_category='vision',
            )
            admin_system_event = main.DetectionEvent(
                user_id=self.admin_id,
                event_type='ADMIN SYSTEM EVENT',
                event_category='system',
            )
            main.db.session.add_all([
                main.DetectionEvent(
                    user_id=self.admin_id,
                    event_type='ADMIN A VISION EVENT',
                    event_category='vision',
                ),
                main.DetectionEvent(
                    user_id=other_admin.id,
                    event_type='ADMIN B VISION EVENT',
                    event_category='vision',
                ),
                user_event,
                admin_system_event,
                main.AuditLog(
                    user_id=self.admin_id,
                    event_type='login',
                    description='Admin logged in',
                ),
            ])
            main.db.session.commit()
            other_admin_id = other_admin.id

        self.login_as(self.admin_id)
        dashboard = self.client.get('/admin')
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b'Clear computer-vision event logs for all admin accounts?', dashboard.data)
        self.assertIn(b'action="/admin/system/clear_events"', dashboard.data)

        response = self.client.post('/admin/system/clear_events')
        self.assertEqual(response.status_code, 302)
        with self.app.app_context():
            remaining_events = main.DetectionEvent.query.order_by(
                main.DetectionEvent.event_type,
            ).all()
            self.assertEqual(
                {(event.user_id, event.event_type) for event in remaining_events},
                {
                    (self.user_id, 'USER VISION EVENT'),
                    (self.admin_id, 'ADMIN SYSTEM EVENT'),
                },
            )
            self.assertNotEqual(self.admin_id, other_admin_id)
            self.assertEqual(
                main.AuditLog.query.filter_by(event_type='login').count(),
                1,
            )
            clear_audit = main.AuditLog.query.filter_by(event_type='clear').one()
            self.assertIn('2 admin computer-vision event log(s)', clear_audit.description)

    def test_camera_settings_update_is_owner_scoped_and_restarts_changed_stream(self):
        payload = {
            'name': 'Updated Front Gate',
            'zone': 'North Entry',
            'source': 'rtsps://example/new-camera',
            'cv_processing_enabled': False,
            'fire_detection_enabled': False,
            'smoke_detection_enabled': None,
            'face_recognition_enabled': True,
            'email_alerts_enabled': True,
            'unidentified_email_alerts_enabled': None,
        }
        self.login_as(self.user_id)
        with patch.object(main, 'release_camera_streams') as release_streams:
            response = self.client.put(
                f'/api/camera/{self.camera_id}/settings',
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        release_streams.assert_called_once_with(self.camera_id)
        with self.app.app_context():
            camera = main.db.session.get(main.Camera, self.camera_id)
            self.assertEqual(camera.name, payload['name'])
            self.assertEqual(camera.zone, payload['zone'])
            self.assertEqual(camera.source, payload['source'])
            self.assertFalse(camera.cv_processing_enabled)
            self.assertFalse(camera.fire_detection_enabled)
            self.assertIsNone(camera.smoke_detection_enabled)
            self.assertTrue(camera.face_recognition_enabled)
            self.assertTrue(camera.email_alerts_enabled)

        self.login_as(self.admin_id)
        denied = self.client.put(
            f'/api/camera/{self.camera_id}/settings',
            json=payload,
        )
        self.assertEqual(denied.status_code, 403)

    def test_camera_processing_overrides_inherit_account_values(self):
        manager = object.__new__(main.VideoStreamManager)
        manager.camera_settings = {
            'cv_processing_enabled': True,
            'fire_detection_enabled': False,
            'smoke_detection_enabled': None,
            'face_recognition_enabled': True,
            'email_alerts_enabled': True,
            'unidentified_email_alerts_enabled': None,
        }
        settings = manager._apply_camera_settings({
            'yolo_enabled': True,
            'face_recognition_enabled': False,
            'active_classes': 'fire,smoke',
            'email_alerts_enabled': False,
            'unidentified_email_alerts_enabled': True,
        })
        self.assertTrue(settings['cv_processing_enabled'])
        self.assertFalse(settings['fire_detection_enabled'])
        self.assertTrue(settings['smoke_detection_enabled'])
        self.assertTrue(settings['face_recognition_enabled'])
        self.assertTrue(settings['email_alerts_enabled'])
        self.assertTrue(settings['unidentified_email_alerts_enabled'])
        self.assertEqual(settings['active_classes'], 'smoke')

    def test_user_camera_pages_render_the_camera_settings_controls(self):
        self.login_as(self.user_id)
        dashboard = self.client.get('/')
        cctv = self.client.get('/cctv')
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b'Edit settings', dashboard.data)
        self.assertEqual(cctv.status_code, 200)
        self.assertIn(b'Edit camera settings', cctv.data)
        self.assertIn(b'cameraSettingsOverlay', cctv.data)

    def test_camera_settings_reject_invalid_stream_url_and_inactive_camera(self):
        self.login_as(self.user_id)
        invalid = self.client.put(
            f'/api/camera/{self.camera_id}/settings',
            json={
                'name': 'Front Gate',
                'zone': 'North Entry',
                'source': '0',
                'cv_processing_enabled': True,
            },
        )
        self.assertEqual(invalid.status_code, 400)

        with self.app.app_context():
            camera = main.db.session.get(main.Camera, self.camera_id)
            camera.archived_at = datetime.utcnow()
            main.db.session.commit()
        archived = self.client.get(f'/api/camera/{self.camera_id}/settings')
        self.assertEqual(archived.status_code, 403)

    def test_user_dashboard_detection_metrics_and_events_are_limited_to_today(self):
        today_start = datetime(2026, 9, 2)
        tomorrow_start = datetime(2026, 9, 3)
        with self.app.app_context():
            main.db.session.add_all([
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    event_type='TODAY HIGH',
                    description='Today high event',
                    event_category='vision',
                    severity='high',
                    timestamp=datetime(2026, 9, 2, 8),
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    event_type='TODAY NORMAL',
                    description='Today normal event',
                    event_category='vision',
                    severity='normal',
                    timestamp=datetime(2026, 9, 2, 9),
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    event_type='YESTERDAY',
                    description='Yesterday event',
                    event_category='vision',
                    severity='critical',
                    timestamp=datetime(2026, 9, 1, 23, 59, 59),
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    camera_id=self.camera_id,
                    event_type='TOMORROW',
                    description='Tomorrow event',
                    event_category='vision',
                    severity='critical',
                    timestamp=tomorrow_start,
                ),
            ])
            main.db.session.commit()

        self.login_as(self.user_id)
        with patch(
            'main.get_manila_day_utc_range',
            return_value=(today_start, tomorrow_start),
        ), patch('main.camera_status', return_value=(False, False)):
            metrics = self.client.get('/api/user/dashboard-metrics')
            dashboard = self.client.get('/')

        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json['total_detections'], 2)
        self.assertEqual(metrics.json['high_severity_alerts'], 1)
        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b'Today high event', dashboard.data)
        self.assertIn(b'Today normal event', dashboard.data)
        self.assertNotIn(b'Yesterday event', dashboard.data)
        self.assertNotIn(b'Tomorrow event', dashboard.data)

    def test_admin_dashboard_alerts_and_activity_are_limited_to_today(self):
        today_start = datetime(2026, 9, 2)
        tomorrow_start = datetime(2026, 9, 3)
        with self.app.app_context():
            main.db.session.add_all([
                main.DetectionEvent(
                    user_id=self.user_id,
                    event_type='TODAY ALERT',
                    description='Today admin alert',
                    event_category='vision',
                    timestamp=datetime(2026, 9, 2, 8),
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    event_type='YESTERDAY ALERT',
                    description='Yesterday admin alert',
                    event_category='vision',
                    timestamp=datetime(2026, 9, 1, 23, 59, 59),
                ),
                main.DetectionEvent(
                    user_id=self.user_id,
                    event_type='TOMORROW ALERT',
                    description='Tomorrow admin alert',
                    event_category='vision',
                    timestamp=tomorrow_start,
                ),
            ])
            main.db.session.commit()

        self.login_as(self.admin_id)
        with patch(
            'main.get_manila_day_utc_range',
            return_value=(today_start, tomorrow_start),
        ):
            dashboard = self.client.get('/admin')

        self.assertEqual(dashboard.status_code, 200)
        self.assertIn(b'Security Alerts Today', dashboard.data)
        self.assertIn(b'id="todaySecurityAlertsCount">1</div>', dashboard.data)
        self.assertIn(b'Today admin alert', dashboard.data)
        self.assertNotIn(b'Yesterday admin alert', dashboard.data)
        self.assertNotIn(b'Tomorrow admin alert', dashboard.data)

    def test_admin_dashboard_camera_counts_only_admin_owned_public_cameras(self):
        with self.app.app_context():
            other_admin = main.User(
                username='camera-admin-b',
                email='camera-admin-b@example.com',
                password='secret',
                role='admin',
            )
            main.db.session.add(other_admin)
            main.db.session.flush()
            main.db.session.add_all([
                main.Camera(
                    user_id=self.admin_id,
                    name='Admin public active',
                    source='rtsp://example/admin-public-active',
                    is_public=True,
                    is_active=True,
                ),
                main.Camera(
                    user_id=self.admin_id,
                    name='Admin public disabled',
                    source='rtsp://example/admin-public-disabled',
                    is_public=True,
                    is_active=False,
                ),
                main.Camera(
                    user_id=self.admin_id,
                    name='Admin private active',
                    source='rtsp://example/admin-private-active',
                    is_public=False,
                    is_active=True,
                ),
                main.Camera(
                    user_id=self.user_id,
                    name='User public active',
                    source='rtsp://example/user-public-active',
                    is_public=True,
                    is_active=True,
                ),
                main.Camera(
                    user_id=other_admin.id,
                    name='Other admin public active',
                    source='rtsp://example/other-admin-public-active',
                    is_public=True,
                    is_active=True,
                ),
            ])
            main.db.session.commit()

        self.login_as(self.admin_id)
        with patch(
            'main.get_manila_day_utc_range',
            return_value=(datetime(2026, 9, 2), datetime(2026, 9, 3)),
        ):
            response = self.client.get('/admin')

        self.assertEqual(response.status_code, 200)
        self.assertIn(b'id="activeCameraFeeds">2</div>', response.data)
        self.assertIn(b'id="cameraCoverageDelta">67</span>% of 3 feeds online', response.data)

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

        with patch(
            'main.get_manila_day_utc_range',
            return_value=(
                datetime(2024, 1, 1, 16),
                datetime(2024, 1, 2, 16),
            ),
        ):
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

    def test_surveillance_report_counts_persisted_connection_transitions(self):
        now = main.datetime.utcnow()
        with self.app.app_context():
            camera = main.Camera(
                user_id=self.admin_id,
                name='Public Gate Camera',
                source='rtsp://example/public-gate',
                zone='Main Entrance',
                is_public=True,
            )
            main.db.session.add(camera)
            main.db.session.commit()

            self.assertTrue(main.record_camera_connection_transition(
                camera.id,
                'connected',
                now - main.timedelta(minutes=60),
            ))
            self.assertTrue(main.record_camera_connection_transition(
                camera.id,
                'disconnected',
                now - main.timedelta(minutes=30),
            ))
            self.assertFalse(main.record_camera_connection_transition(
                camera.id,
                'disconnected',
                now - main.timedelta(minutes=25),
            ))
            self.assertTrue(main.record_camera_connection_transition(
                camera.id,
                'connected',
                now - main.timedelta(minutes=10),
            ))

            report = main.build_report(
                main.DetectionEvent,
                report_type='surveillance',
                timeframe='24h',
                now=now,
                camera_model=main.Camera,
                connection_event_model=main.CameraConnectionEvent,
                camera_status_getter=lambda _camera: {
                    'connected': True,
                    'recording': False,
                },
            )
            csv_report = main.generate_report_csv(report)

        self.assertEqual(len(report['surveillance_summary']), 1)
        row = report['surveillance_summary'][0]
        self.assertEqual(row['status'], 'Connected')
        self.assertEqual(row['recording_status'], 'Not recording')
        self.assertEqual(row['disconnect_events'], 1)
        self.assertEqual(row['reconnect_events'], 1)
        self.assertEqual(row['offline_duration'], '20m')
        self.assertEqual(row['availability'], '66.7%')
        self.assertNotEqual(row['last_state_change'], 'No history')
        self.assertIn('Observed Availability', csv_report)

        self.login_as(self.admin_id)
        response = self.client.get('/admin/reports?type=surveillance&timeframe=24h')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Cameras Monitored', response.data)
        self.assertIn(b'Observed Downtime', response.data)
        self.assertIn(b'Reconnects', response.data)

    def test_live_camera_status_persists_only_actual_state_changes(self):
        with self.app.app_context():
            camera = main.Camera(
                user_id=self.admin_id,
                name='Monitored Public Camera',
                source='rtsp://example/monitored',
                is_public=True,
            )
            main.db.session.add(camera)
            main.db.session.commit()
            camera_id = camera.id

        class FakeManager:
            is_recording = False

            def __init__(self, connected):
                self.connected = connected

            def is_connected(self):
                return self.connected

        managers = main.active_user_streams.setdefault(self.admin_id, {})
        managers[camera_id] = FakeManager(True)
        try:
            with self.app.app_context():
                camera = main.db.session.get(main.Camera, camera_id)
                self.assertTrue(main.camera_live_status(camera)['connected'])
                self.assertTrue(main.camera_live_status(camera)['connected'])
                managers[camera_id].connected = False
                self.assertFalse(main.camera_live_status(camera)['connected'])
                statuses = [
                    event.status
                    for event in main.CameraConnectionEvent.query.filter_by(
                        camera_id=camera_id,
                    ).order_by(main.CameraConnectionEvent.id).all()
                ]
                self.assertEqual(statuses, ['connected', 'disconnected'])
        finally:
            managers.pop(camera_id, None)
            if not managers:
                main.active_user_streams.pop(self.admin_id, None)

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

    def test_admin_report_snapshot_opens_from_modal_control(self):
        with self.app.app_context():
            camera = main.Camera(
                user_id=self.admin_id,
                name='Report Snapshot Camera',
                source='rtsp://example/report-snapshot',
                is_public=True,
            )
            main.db.session.add(camera)
            main.db.session.flush()
            event = main.DetectionEvent(
                user_id=self.admin_id,
                camera_id=camera.id,
                source_name=camera.name,
                event_type='SNAPSHOT TEST EVENT',
                event_category='vision',
                severity='high',
                snapshot_path='123.jpg',
                timestamp=datetime.utcnow(),
            )
            main.db.session.add(event)
            main.db.session.commit()
            event_id = event.id

        self.login_as(self.admin_id)
        response = self.client.get('/admin/reports?timeframe=24h')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'data-report-snapshot="/admin/events/', response.data)
        self.assertIn(f'/admin/events/{event_id}/snapshot'.encode(), response.data)
        self.assertIn(b'id="reportSnapshotModal"', response.data)
        self.assertIn(b'id="reportSnapshotImage"', response.data)

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

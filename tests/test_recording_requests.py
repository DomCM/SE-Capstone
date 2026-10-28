import base64
import io
import json
import os
import tempfile
import unittest

import main
from main import app, db, User, Camera, RecordingRequest
from media_storage import MediaStorage


class RecordingRequestTests(unittest.TestCase):
    def setUp(self):
        app.config.update(
            TESTING=True,
            SECRET_KEY='test-secret',
            SQLALCHEMY_DATABASE_URI='sqlite:///:memory:',
            SQLALCHEMY_TRACK_MODIFICATIONS=False,
        )
        with app.app_context():
            db.drop_all()
            db.create_all()
            user = User(username='user', email='user@example.com', password='secret', role='user')
            admin = User(username='admin', email='admin@example.com', password='secret', role='admin')
            db.session.add_all([user, admin])
            db.session.flush()
            db.session.add(Camera(user_id=user.id, name='Front camera', source='rtsp://front'))
            db.session.add(Camera(user_id=admin.id, name='Public TEST', source='rtsp://public', is_public=True))
            db.session.commit()
            self.user_id = user.id
            self.admin_id = admin.id
            self.camera_id = user.cameras[0].id
            self.public_camera_id = Camera.query.filter_by(name='Public TEST').one().id
        self.recordings_dir = tempfile.TemporaryDirectory()
        self.previous_recordings_dir = main.BASE_RECORDINGS_DIR
        self.previous_media_storage = main.media_storage
        main.BASE_RECORDINGS_DIR = self.recordings_dir.name
        test_key = base64.urlsafe_b64encode(bytes(range(32))).decode('ascii')
        main.media_storage = MediaStorage(test_key)
        self.client = app.test_client()

    def tearDown(self):
        main.BASE_RECORDINGS_DIR = self.previous_recordings_dir
        main.media_storage = self.previous_media_storage
        self.recordings_dir.cleanup()

    def login_as(self, user_id):
        with self.client.session_transaction() as session:
            session['privacy_agreed'] = True
            session['_user_id'] = str(user_id)
            session['_fresh'] = True

    def test_user_request_admin_fulfillment_and_download(self):
        self.login_as(self.user_id)
        response = self.client.post('/recording_requests', data={
            'camera_id': self.public_camera_id,
            'date_needed': '2026-09-04',
            'start_time': '09:00',
            'end_time': '10:00',
            'reason': 'Incident review',
        })
        self.assertEqual(response.status_code, 302)

        with app.app_context():
            recording_request = RecordingRequest.query.one()
            request_id = recording_request.id
            self.assertEqual(recording_request.status, 'pending')
            self.assertEqual(recording_request.time_range, '09:00 - 10:00')
            self.assertEqual(recording_request.camera_id, self.public_camera_id)

        self.login_as(self.admin_id)
        self.assertEqual(self.client.get('/admin/request').status_code, 200)
        self.assertEqual(self.client.post(f'/admin/request/{request_id}/approve').status_code, 302)
        response = self.client.post(
            f'/admin/request/{request_id}/upload',
            data={'video_file': (io.BytesIO(b'clip data'), 'clip.mp4')},
            content_type='multipart/form-data',
        )
        self.assertEqual(response.status_code, 302)

        with app.app_context():
            recording_request = db.session.get(RecordingRequest, request_id)
            self.assertEqual(recording_request.status, 'fulfilled')
            self.assertTrue(os.path.isfile(recording_request.video_path + '.enc'))

        self.login_as(self.user_id)
        request_page = self.client.get('/user_request')
        self.assertEqual(request_page.status_code, 200)
        self.assertIn(b'View Footage', request_page.data)

        preview = self.client.get(f'/recording_requests/{request_id}/view')
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.data, b'clip data')
        self.assertNotIn('attachment', preview.headers.get('Content-Disposition', ''))
        preview.close()

        ranged_preview = self.client.get(
            f'/recording_requests/{request_id}/view',
            headers={'Range': 'bytes=1-3'},
        )
        self.assertEqual(ranged_preview.status_code, 206)
        self.assertEqual(ranged_preview.data, b'lip')
        ranged_preview.close()

        response = self.client.get(f'/recording_requests/{request_id}/download')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data, b'clip data')
        response.close()

        other_user_id = None
        with app.app_context():
            other_user = User(username='other', email='other@example.com', password='secret', role='user')
            db.session.add(other_user)
            db.session.commit()
            other_user_id = other_user.id
        self.login_as(other_user_id)
        unauthorized_preview = self.client.get(f'/recording_requests/{request_id}/view')
        self.assertEqual(unauthorized_preview.status_code, 404)

    def test_public_camera_is_available_for_recording_requests(self):
        self.login_as(self.user_id)

        page = self.client.get('/user_request')
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'Public TEST', page.data)
        self.assertNotIn(b'Front camera</option>', page.data)

        response = self.client.post('/recording_requests', data={
            'camera_id': self.public_camera_id,
            'date_needed': '2026-09-04',
            'start_time': '09:00',
            'end_time': '10:00',
            'reason': 'Review public camera footage',
        })
        self.assertEqual(response.status_code, 302)
        with app.app_context():
            recording_request = RecordingRequest.query.one()
            self.assertEqual(recording_request.camera_id, self.public_camera_id)
            self.assertEqual(recording_request.user_id, self.user_id)

    def test_private_camera_cannot_be_requested_by_direct_submission(self):
        self.login_as(self.user_id)

        response = self.client.post('/recording_requests', data={
            'camera_id': self.camera_id,
            'date_needed': '2026-09-04',
            'start_time': '09:00',
            'end_time': '10:00',
            'reason': 'Review private camera footage',
        })

        self.assertEqual(response.status_code, 302)
        with app.app_context():
            self.assertEqual(RecordingRequest.query.count(), 0)

    def test_archiving_camera_preserves_request_history_and_can_be_restored(self):
        with app.app_context():
            recording_request = RecordingRequest(
                user_id=self.user_id,
                camera_id=self.camera_id,
                camera_name='Front camera',
                date_needed='2026-09-04',
                time_range='09:00 - 10:00',
                reason='Incident review',
            )
            db.session.add(recording_request)
            db.session.commit()
            request_id = recording_request.id

        self.login_as(self.user_id)
        response = self.client.post(f'/delete_camera/{self.camera_id}')
        self.assertEqual(response.status_code, 302)
        with app.app_context():
            camera = db.session.get(Camera, self.camera_id)
            recording_request = db.session.get(RecordingRequest, request_id)
            self.assertIsNotNone(camera.archived_at)
            self.assertEqual(recording_request.camera.id, self.camera_id)
            self.assertEqual(recording_request.camera_name, 'Front camera')

        self.assertEqual(self.client.get(f'/video_feed/{self.camera_id}').status_code, 404)
        request_page = self.client.get('/user_request')
        self.assertEqual(request_page.status_code, 200)
        self.assertIn(b'Front camera', request_page.data)
        self.assertNotIn(f'<option value="{self.camera_id}">Front camera</option>'.encode(), request_page.data)

        self.login_as(self.admin_id)
        response = self.client.get('/admin/request')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Front camera', response.data)
        response = self.client.get('/admin/cctv?include_archived=1')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Camera Archived', response.data)
        response = self.client.post(f'/restore_camera/{self.camera_id}')
        self.assertEqual(response.status_code, 302)
        with app.app_context():
            camera = db.session.get(Camera, self.camera_id)
            self.assertIsNone(camera.archived_at)
            self.assertEqual(db.session.get(RecordingRequest, request_id).camera.id, self.camera_id)

    def test_legacy_orphaned_request_does_not_crash_request_pages(self):
        with app.app_context():
            recording_request = RecordingRequest(
                user_id=self.user_id,
                camera_id=999,
                date_needed='2026-09-04',
                time_range='09:00 - 10:00',
                reason='Legacy request',
            )
            db.session.add(recording_request)
            db.session.commit()

        self.login_as(self.user_id)
        user_response = self.client.get('/user_request')
        self.assertEqual(user_response.status_code, 200)
        self.assertIn(b'Camera removed', user_response.data)

        self.login_as(self.admin_id)
        admin_response = self.client.get('/admin/request')
        self.assertEqual(admin_response.status_code, 200)
        self.assertIn(b'Camera removed', admin_response.data)

    def test_recording_request_rejects_malformed_or_reversed_times(self):
        self.login_as(self.user_id)
        invalid_ranges = [
            ('15:00', '14:00', 'End time must be later than start time.'),
            ('not a time', '16:00', 'Please provide valid start and end times.'),
        ]

        for start_time, end_time, message in invalid_ranges:
            with self.subTest(start_time=start_time, end_time=end_time):
                response = self.client.post('/recording_requests', data={
                    'camera_id': self.public_camera_id,
                    'date_needed': '2026-09-04',
                    'start_time': start_time,
                    'end_time': end_time,
                    'reason': 'Incident review',
                }, follow_redirects=True)

                self.assertEqual(response.status_code, 200)
                self.assertIn(message.encode(), response.data)
                with app.app_context():
                    self.assertEqual(RecordingRequest.query.count(), 0)

    def test_admin_can_delete_individual_and_all_virtual_lines(self):
        with app.app_context():
            camera = db.session.get(Camera, self.public_camera_id)
            camera.virtual_lines = json.dumps([
                {'id': 'line-1', 'name': 'Entry', 'points': [[0.1, 0.2], [0.8, 0.7]]},
                {'id': 'line-2', 'name': 'Exit', 'points': [[0.2, 0.3], [0.9, 0.6]]},
            ])
            db.session.commit()

        self.login_as(self.admin_id)
        response = self.client.delete(
            f'/api/camera/{self.public_camera_id}/virtual_lines',
            json={'line_id': 'line-1'},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual([line['id'] for line in response.get_json()['lines']], ['line-2'])

        response = self.client.delete(
            f'/api/camera/{self.public_camera_id}/virtual_lines',
            json={'clear_all': True},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()['lines'], [])

if __name__ == '__main__':
    unittest.main()
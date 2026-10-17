import os
import sys
import threading
from datetime import datetime

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import concealment_detection as cues
import main
from reports import compute_patrol_priority_summary


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeCoordinates:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FakeBox:
    def __init__(self, class_id, confidence):
        self.cls = [FakeScalar(class_id)]
        self.conf = [FakeScalar(confidence)]


class FakePredictionModel:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, **kwargs):
        return [type('Result', (), {
            'names': dict(enumerate(cues.CUE_PROMPTS)),
            'boxes': self.boxes,
        })()]


def test_optional_threshold_is_disabled_until_configured():
    assert cues.parse_optional_threshold(None) is None
    assert cues.parse_optional_threshold('  ') is None
    assert cues.parse_optional_threshold('0.74') == 0.74

    with pytest.raises(ValueError, match='CONCEALMENT_CUE_THRESHOLD'):
        cues.parse_optional_threshold('1.1')


def test_person_crop_requires_both_cues_to_meet_configured_threshold():
    crop = np.zeros((100, 80, 3), dtype=np.uint8)
    model = FakePredictionModel([
        FakeBox(0, 0.81),
        FakeBox(8, 0.73),
    ])

    result = cues.scan_person_crop_with_model(model, crop, 0.70)

    assert result.review_required is True
    assert result.cues == ('face_covering', 'headwear')
    assert result.combined_score == pytest.approx(0.73)

    one_cue_model = FakePredictionModel([
        FakeBox(0, 0.81),
        FakeBox(8, 0.69),
    ])
    one_cue_result = cues.scan_person_crop_with_model(one_cue_model, crop, 0.70)
    assert one_cue_result.review_required is False
    assert one_cue_result.combined_score is None


def test_background_scanner_is_lazy_single_worker_and_drops_queue_overflow():
    inference_started = threading.Event()
    resume_inference = threading.Event()
    factory_calls = []
    callbacks = []
    callback_lock = threading.Lock()

    class BlockingModel:
        def predict(self, **kwargs):
            inference_started.set()
            assert resume_inference.wait(3)
            return [type('Result', (), {'names': {}, 'boxes': []})()]

    def model_factory(model_path, *, prompts):
        factory_calls.append((model_path, prompts))
        return BlockingModel()

    def callback(result, error):
        with callback_lock:
            callbacks.append((result, error))

    scanner = cues.BoundedConcealmentScanner(
        max_queue_size=1,
        scan_cooldown_seconds=30,
        model_factory=model_factory,
        worker_priority=lambda: None,
    )
    crop = np.zeros((64, 48, 3), dtype=np.uint8)
    assert factory_calls == []
    assert scanner.submit(('camera', 1), crop, 0.7, callback) is True
    assert inference_started.wait(3)
    assert scanner.submit(('camera', 2), crop, 0.7, callback) is True
    assert scanner.submit(('camera', 3), crop, 0.7, callback) is False
    assert scanner.submit(('camera', 1), crop, 0.7, callback) is False
    assert len(factory_calls) == 1

    resume_inference.set()
    scanner._queue.join()

    assert len(callbacks) == 2
    assert all(error is None for _, error in callbacks)
    assert scanner.submit(('camera', 1), crop, 0.7, callback) is False
    camera_status = scanner.camera_status('camera')
    assert camera_status['submitted'] == 2
    assert camera_status['skipped_pending'] == 1
    assert camera_status['skipped_queue_full'] == 1
    assert camera_status['completed'] == 2
    assert camera_status['no_combined_cues'] == 2


def test_service_status_reports_disabled_candidates_and_mail_suppression():
    scanner = cues.BoundedConcealmentScanner(
        model_factory=lambda *args, **kwargs: None,
        worker_priority=lambda: None,
    )
    scanner.note_candidate('unknown', threshold_disabled=True)
    scanner.record_notification('suppressed', 'unidentified_email_preference_disabled')

    status = scanner.status()

    assert status['candidates_unknown_face'] == 0
    assert status['candidates_threshold_disabled'] == 1
    assert status['last_candidate']['identity_status'] == 'unknown'
    assert status['last_candidate']['threshold_disabled'] is True
    assert status['notifications_suppressed'] == 1
    assert status['last_notification']['reason'] == 'unidentified_email_preference_disabled'


def test_admin_concealment_diagnostics_api_excludes_smtp_secrets(monkeypatch):
    monkeypatch.setitem(main.app.config, 'CONCEALMENT_CUE_THRESHOLD', 0.65)
    monkeypatch.setitem(main.app.config, 'SMTP_USERNAME', 'smtp-secret-user')
    monkeypatch.setitem(main.app.config, 'SMTP_PASSWORD', 'smtp-secret-password')
    route_function = main.admin_concealment_status_api.__wrapped__

    with main.app.app_context():
        response = route_function()

    payload = response.get_json()
    assert payload['enabled'] is True
    assert payload['threshold'] == 0.65
    assert payload['smtp_username_configured'] is True
    assert payload['smtp_password_configured'] is True
    assert 'smtp-secret-user' not in response.get_data(as_text=True)
    assert 'smtp-secret-password' not in response.get_data(as_text=True)


@pytest.mark.parametrize(
    ('results', 'expected'),
    [
        ([], 'unavailable'),
        ([(1, 2, 3, 4, 'Unknown')], 'unknown'),
        ([(1, 2, 3, 4, 'Known Resident')], None),
    ],
)
def test_only_unusable_or_unknown_faces_trigger_background_scans(results, expected):
    assert main.person_identity_status_for_cue_scan(results) == expected


@pytest.mark.parametrize('camera_is_public', [True, False])
def test_qualified_cues_log_event_with_scores_and_email_only_for_private_camera(
    monkeypatch,
    tmp_path,
    camera_is_public,
):
    monkeypatch.setattr(main, 'BASE_RECORDINGS_DIR', str(tmp_path))
    manager = object.__new__(main.VideoStreamManager)
    manager.camera_id = 14
    manager.user_id = 9
    manager.camera_name = 'Front camera'
    manager.camera_zone = 'Entry'
    manager.camera_is_public = camera_is_public
    events = []
    emails = []
    manager.log_db_event = lambda *args, **kwargs: events.append((args, kwargs))
    manager._send_unidentified_person_email = lambda *args: emails.append(args)
    result = cues.PersonCueResult(0.86, 0.78, 0.70, True)
    observed_at = datetime(2026, 10, 17, 1, 0)

    manager._handle_concealment_scan_result(
        result,
        None,
        identity_status='unknown',
        track_id=4,
        person_crop=np.zeros((60, 40, 3), dtype=np.uint8),
        observed_at=observed_at,
    )

    assert len(events) == 1
    event_args, event_kwargs = events[0]
    assert event_args[0] == 'UNIDENTIFIED PERSON - FACE COVERING + HEADWEAR'
    assert event_kwargs['confidence'] == pytest.approx(0.78)
    assert event_kwargs['detector'] == 'concealment'
    assert event_kwargs['event_metadata']['appearance_cues'] == [
        'face_covering',
        'headwear',
    ]
    assert event_kwargs['event_metadata']['cue_threshold'] == 0.70
    assert event_kwargs['event_metadata']['detected_at'] == observed_at.isoformat()
    assert event_kwargs['event_timestamp'] == observed_at
    assert len(emails) == (0 if camera_is_public else 1)
    scan_files = list((tmp_path / '9' / 'image_scans').iterdir())
    assert len(scan_files) == 3
    assert {path.suffix for path in scan_files} == {'.jpg', '.txt'}
    report = next(path for path in scan_files if path.suffix == '.txt').read_text()
    assert 'Outcome: review required' in report
    assert 'Face-covering score: 0.860' in report


def test_public_concealment_events_remain_eligible_for_capped_admin_analytics():
    event = {
        'camera_id': 22,
        'camera_is_public': True,
        'zone': 'Public entrance',
        'camera_name': 'Gate camera',
        'event_type': 'UNIDENTIFIED PERSON - FACE COVERING + HEADWEAR',
        'raw_event_type': 'UNIDENTIFIED PERSON - FACE COVERING + HEADWEAR',
        'detector': 'concealment',
        'object_class': 'person',
        'severity': 'Warning',
        'raw_timestamp': datetime(2026, 10, 17, 1, 0),
    }

    summary = compute_patrol_priority_summary([event])

    assert summary[0]['unknown_faces'] == 1
    assert summary[0]['risk_score'] == 1
    assert summary[0]['top_event'] == 'UNIDENTIFIED PERSON - FACE COVERING + HEADWEAR'

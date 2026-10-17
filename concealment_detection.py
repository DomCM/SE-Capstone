"""Reusable, low-priority appearance-cue detector and asynchronous worker."""

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import math
import os
from pathlib import Path
import queue
import threading
import time
from typing import Callable, Hashable, Optional, Tuple

import cv2


PROMPTS = (
    'person',
    'face mask',
    'surgical mask',
    'cloth face covering',
    'balaclava',
    'ski mask',
    'face covering',
    'person wearing a balaclava',
    'person wearing a ski mask',
    'baseball cap',
    'cap',
    'hat',
    'beanie',
    'person wearing a face covering',
)
FACE_COVERING_CLASSES = {
    'face mask',
    'surgical mask',
    'cloth face covering',
    'balaclava',
    'ski mask',
    'face covering',
    'person wearing a balaclava',
    'person wearing a ski mask',
    'person wearing a face covering',
}
HEADWEAR_CLASSES = {'baseball cap', 'cap', 'hat', 'beanie'}
CUE_PROMPTS = tuple(prompt for prompt in PROMPTS if prompt != 'person')
DEFAULT_MODEL = 'yoloe-26s-seg.pt'
DEFAULT_QUEUE_SIZE = 8
DEFAULT_SCAN_COOLDOWN_SECONDS = 45.0
DEFAULT_IMAGE_SIZE = 480
DEFAULT_IMAGE_CONFIDENCE = 0.70
DEFAULT_CUE_THRESHOLD = 0.70

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CueAssessment:
    category: str
    review_required: bool
    cues: Tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class PersonImageAssessment:
    box: Tuple[int, int, int, int]
    face_covering_score: Optional[float]
    headwear_score: Optional[float]
    category: str
    review_required: bool
    cues: Tuple[str, ...]
    explanation: str


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    box: Tuple[float, float, float, float]


@dataclass(frozen=True)
class PersonCueResult:
    face_covering_score: Optional[float]
    headwear_score: Optional[float]
    cue_threshold: float
    review_required: bool

    @property
    def cues(self):
        return tuple(
            cue for cue, score in (
                ('face_covering', self.face_covering_score),
                ('headwear', self.headwear_score),
            )
            if score is not None and score >= self.cue_threshold
        )

    @property
    def combined_score(self):
        if not self.review_required:
            return None
        return min(self.face_covering_score, self.headwear_score)


@dataclass(frozen=True)
class _ScanJob:
    key: Hashable
    image: object
    cue_threshold: float
    callback: Callable[[Optional[PersonCueResult], Optional[BaseException]], None]


def validate_threshold(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number between 0 and 1')
    if not math.isfinite(value) or value < 0 or value > 1:
        raise ValueError(f'{name} must be a finite number between 0 and 1')
    return float(value)


def parse_optional_threshold(value):
    """Return a validated threshold, or None when automatic cue alerts are disabled."""
    if value is None or not str(value).strip():
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError('CONCEALMENT_CUE_THRESHOLD must be a number between 0 and 1') from exc
    try:
        return validate_threshold('CONCEALMENT_CUE_THRESHOLD', parsed)
    except ValueError as exc:
        raise ValueError('CONCEALMENT_CUE_THRESHOLD must be a finite number between 0 and 1') from exc


def _validate_score(name: str, score: Optional[float]) -> Optional[float]:
    if score is None:
        return None
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f'{name} must be a number between 0 and 1, or None')
    if not math.isfinite(score) or score < 0 or score > 1:
        raise ValueError(f'{name} must be a finite number between 0 and 1')
    return float(score)


def assess_person_cues(
    *,
    face_assessable: bool,
    face_covering_score: Optional[float],
    headwear_score: Optional[float],
    identity_recognized: Optional[bool] = None,
    cue_threshold: float = 0.70,
) -> CueAssessment:
    """Interpret appearance cues without inferring a person's intent."""
    if not isinstance(face_assessable, bool):
        raise ValueError('face_assessable must be a boolean')
    if identity_recognized is not None and not isinstance(identity_recognized, bool):
        raise ValueError('identity_recognized must be a boolean or None')
    cue_threshold = validate_threshold('cue_threshold', cue_threshold)
    face_covering_score = _validate_score('face_covering_score', face_covering_score)
    headwear_score = _validate_score('headwear_score', headwear_score)
    face_covering_observed = (
        face_covering_score is not None and face_covering_score >= cue_threshold
    )
    headwear_observed = headwear_score is not None and headwear_score >= cue_threshold
    cues = tuple(
        cue
        for cue, observed in (
            ('face_covering', face_covering_observed),
            ('headwear', headwear_observed),
        )
        if observed
    )

    if face_covering_observed and headwear_observed:
        return CueAssessment(
            category='multiple_cues_for_review',
            review_required=True,
            cues=cues,
            explanation=(
                'Face covering and headwear cues were both detected. '
                'This is a review cue only; intent is unknown.'
            ),
        )
    if cues:
        return CueAssessment(
            category='single_cue_observed',
            review_required=True,
            cues=cues,
            explanation=(
                'One appearance cue was detected. Human review is recommended; '
                'this is not a combined-cue result and intent is unknown.'
            ),
        )
    if not face_assessable:
        return CueAssessment(
            category='face_not_assessable',
            review_required=False,
            cues=(),
            explanation=(
                'The face could not be assessed. Distance, pose, lighting, blur, '
                'or occlusion may explain this; no selected cue was detected.'
            ),
        )
    if identity_recognized is True:
        category = 'recognized_face_no_cues'
        explanation = 'The face was assessed and recognized; no selected appearance cues were detected.'
    elif identity_recognized is False:
        category = 'unrecognized_face_no_cues'
        explanation = 'The face was assessable but not recognized; no selected appearance cues were detected.'
    else:
        category = 'face_assessed_identity_not_checked'
        explanation = 'The face was assessable, but identity recognition was not evaluated.'
    return CueAssessment(category, False, (), explanation)


def _normalize_class_name(name: str) -> str:
    return ' '.join(str(name).lower().replace('_', ' ').split())


def _overlap_ratio(inner_box, outer_box):
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    intersection_width = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    intersection_height = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area == 0:
        return 0.0
    return intersection_width * intersection_height / inner_area


def associate_cues_with_people(detections, cue_threshold=DEFAULT_CUE_THRESHOLD):
    """Assign accessory detections to person boxes and assess each person."""
    cue_threshold = validate_threshold('cue_threshold', cue_threshold)
    people = [item for item in detections if item.class_name in {'person'}]
    face_coverings = [
        item for item in detections
        if item.class_name in FACE_COVERING_CLASSES and item.confidence >= cue_threshold
    ]
    headwear = [
        item for item in detections
        if item.class_name in HEADWEAR_CLASSES and item.confidence >= cue_threshold
    ]

    assessments = []
    for person in people:
        person_coverings = [
            item.confidence for item in face_coverings
            if _overlap_ratio(item.box, person.box) >= 0.5
        ]
        person_headwear = [
            item.confidence for item in headwear
            if _overlap_ratio(item.box, person.box) >= 0.5
        ]
        face_covering_score = max(person_coverings, default=None)
        headwear_score = max(person_headwear, default=None)
        cues = tuple(
            label for label, score in (
                ('face_covering', face_covering_score),
                ('headwear', headwear_score),
            )
            if score is not None
        )
        both_cues = len(cues) == 2
        if both_cues:
            category = 'multiple_cues_for_review'
            explanation = (
                'Face covering and headwear detections overlap this person. '
                'Human review is required; intent is unknown.'
            )
        elif cues:
            category = 'single_cue_observed'
            explanation = (
                'One appearance cue was detected. Human review is recommended; '
                'this is not a combined-cue result and intent is unknown.'
            )
        else:
            category = 'no_combined_cues_detected'
            explanation = (
                'No combined face-covering and headwear cues were detected. '
                'This does not establish whether the face is identifiable.'
            )
        assessments.append(PersonImageAssessment(
            box=tuple(int(round(value)) for value in person.box),
            face_covering_score=face_covering_score,
            headwear_score=headwear_score,
            category=category,
            review_required=bool(cues),
            cues=cues,
            explanation=explanation,
        ))
    return assessments


def _load_detections(model, image, confidence):
    result = model.predict(
        source=image,
        conf=0.001,
        imgsz=640,
        device='cpu',
        verbose=False,
    )[0]
    detections = []
    if result.boxes is None:
        return detections

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = _normalize_class_name(result.names[class_id])
        confidence_score = float(box.conf[0].item())
        coordinates = tuple(float(value) for value in box.xyxy[0].tolist())
        detections.append(Detection(class_name, confidence_score, coordinates))
    return detections


def load_yoloe_model(model_path=DEFAULT_MODEL, prompts=PROMPTS):
    try:
        import torch
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError(
            'Ultralytics is required for image scanning; install project requirements first.'
        ) from exc
    torch.set_num_threads(1)
    model = YOLOE(str(model_path))
    model.set_classes(list(prompts))
    return model


def scan_person_crop_with_model(model, image, cue_threshold, image_size=DEFAULT_IMAGE_SIZE):
    """Score face-covering and headwear cues on a single detected person's crop."""
    cue_threshold = validate_threshold('cue_threshold', cue_threshold)
    if image is None or not getattr(image, 'size', 0):
        raise ValueError('person crop must be a non-empty image')

    result = model.predict(
        source=image,
        conf=cue_threshold,
        imgsz=image_size,
        device='cpu',
        verbose=False,
    )[0]
    face_covering_score = None
    headwear_score = None
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            class_name = _normalize_class_name(result.names[class_id])
            confidence = float(box.conf[0].item())
            if class_name in FACE_COVERING_CLASSES:
                face_covering_score = max(face_covering_score or 0.0, confidence)
            elif class_name in HEADWEAR_CLASSES:
                headwear_score = max(headwear_score or 0.0, confidence)

    review_required = (
        face_covering_score is not None
        and face_covering_score >= cue_threshold
        and headwear_score is not None
        and headwear_score >= cue_threshold
    )
    return PersonCueResult(
        face_covering_score=face_covering_score,
        headwear_score=headwear_score,
        cue_threshold=cue_threshold,
        review_required=review_required,
    )


def _lower_current_thread_priority():
    """Best-effort lower priority for the inference worker thread only."""
    try:
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            current_thread = kernel32.GetCurrentThread()
            if not kernel32.SetThreadPriority(current_thread, -1):
                raise ctypes.WinError(ctypes.get_last_error())
        elif hasattr(os, 'setpriority') and hasattr(os, 'PRIO_PROCESS'):
            os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except (AttributeError, OSError) as exc:
        logger.warning('Could not lower concealment worker priority: %s', exc)


class BoundedConcealmentScanner:
    """Single-worker, bounded queue that keeps model inference off video threads."""

    def __init__(
        self,
        *,
        model_path=DEFAULT_MODEL,
        max_queue_size=DEFAULT_QUEUE_SIZE,
        scan_cooldown_seconds=DEFAULT_SCAN_COOLDOWN_SECONDS,
        image_size=DEFAULT_IMAGE_SIZE,
        model_factory: Optional[Callable] = None,
        worker_priority: Optional[Callable] = None,
    ):
        if isinstance(max_queue_size, bool) or not isinstance(max_queue_size, int) or max_queue_size < 1:
            raise ValueError('max_queue_size must be a positive integer')
        if (
            isinstance(scan_cooldown_seconds, bool)
            or not isinstance(scan_cooldown_seconds, (int, float))
            or not math.isfinite(scan_cooldown_seconds)
            or scan_cooldown_seconds < 0
        ):
            raise ValueError('scan_cooldown_seconds must be a finite non-negative number')
        self.model_path = model_path
        self.image_size = image_size
        self.scan_cooldown_seconds = float(scan_cooldown_seconds)
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._lock = threading.Lock()
        self._pending = set()
        self._last_submitted = {}
        self._worker = None
        self._model = None
        self._model_factory = model_factory or load_yoloe_model
        self._worker_priority = worker_priority or _lower_current_thread_priority
        self._metrics = {
            'candidates_no_usable_face': 0,
            'candidates_unknown_face': 0,
            'candidates_threshold_disabled': 0,
            'submitted': 0,
            'skipped_pending': 0,
            'skipped_cooldown': 0,
            'skipped_queue_full': 0,
            'started': 0,
            'completed': 0,
            'failed': 0,
            'combined_cues_found': 0,
            'no_combined_cues': 0,
            'notifications_suppressed': 0,
            'notifications_queued': 0,
            'notifications_sent': 0,
            'notification_failures': 0,
        }
        self._last_scan = None
        self._last_notification = None
        self._last_scan_started_at = None
        self._active_scan_started_at = None
        self._last_candidate = None
        self._camera_metrics = {}

    @property
    def queue_size(self):
        return self._queue.qsize()

    def _new_camera_metrics(self):
        return {
            'metrics': {
                'candidates_no_usable_face': 0,
                'candidates_unknown_face': 0,
                'candidates_threshold_disabled': 0,
                'submitted': 0,
                'skipped_pending': 0,
                'skipped_cooldown': 0,
                'skipped_queue_full': 0,
                'started': 0,
                'completed': 0,
                'failed': 0,
                'combined_cues_found': 0,
                'no_combined_cues': 0,
                'notifications_suppressed': 0,
                'notifications_queued': 0,
                'notifications_sent': 0,
                'notification_failures': 0,
            },
            'last_scan': None,
            'last_candidate': None,
            'last_notification': None,
        }

    def _camera_record(self, camera_id):
        if camera_id is None:
            return None
        return self._camera_metrics.setdefault(camera_id, self._new_camera_metrics())

    def _increment(self, metric, camera_id=None):
        self._metrics[metric] += 1
        record = self._camera_record(camera_id)
        if record is not None:
            record['metrics'][metric] += 1

    def note_candidate(self, identity_status, *, camera_id=None, threshold_disabled=False):
        with self._lock:
            if threshold_disabled:
                self._increment('candidates_threshold_disabled', camera_id)
            elif identity_status == 'unknown':
                self._increment('candidates_unknown_face', camera_id)
            elif identity_status == 'unavailable':
                self._increment('candidates_no_usable_face', camera_id)
            candidate = {
                'identity_status': identity_status,
                'threshold_disabled': threshold_disabled,
                'at': datetime.now(timezone.utc).isoformat(),
            }
            self._last_candidate = candidate
            record = self._camera_record(camera_id)
            if record is not None:
                record['last_candidate'] = candidate.copy()

    def record_notification(self, outcome, reason=None, *, camera_id=None):
        counter = {
            'suppressed': 'notifications_suppressed',
            'queued': 'notifications_queued',
            'sent': 'notifications_sent',
            'failed': 'notification_failures',
        }.get(outcome)
        if counter is None:
            raise ValueError(f'Unknown notification outcome: {outcome}')
        with self._lock:
            self._increment(counter, camera_id)
            notification = {
                'outcome': outcome,
                'reason': reason,
                'at': datetime.now(timezone.utc).isoformat(),
            }
            self._last_notification = notification
            record = self._camera_record(camera_id)
            if record is not None:
                record['last_notification'] = notification.copy()

    def status(self):
        with self._lock:
            return {
                **self._metrics,
                'model_path': str(self.model_path),
                'model_loaded': self._model is not None,
                'worker_alive': bool(self._worker and self._worker.is_alive()),
                'scan_in_progress': self._active_scan_started_at is not None,
                'active_scan_started_at': self._active_scan_started_at,
                'queue_size': self._queue.qsize(),
                'queue_capacity': self._queue.maxsize,
                'last_scan': self._last_scan.copy() if self._last_scan else None,
                'last_candidate': self._last_candidate.copy() if self._last_candidate else None,
                'last_notification': (
                    self._last_notification.copy() if self._last_notification else None
                ),
                'cameras': {
                    str(camera_id): {
                        **record['metrics'],
                        'last_scan': (
                            record['last_scan'].copy() if record['last_scan'] else None
                        ),
                        'last_candidate': (
                            record['last_candidate'].copy()
                            if record['last_candidate'] else None
                        ),
                        'last_notification': (
                            record['last_notification'].copy()
                            if record['last_notification'] else None
                        ),
                    }
                    for camera_id, record in self._camera_metrics.items()
                },
            }

    def camera_status(self, camera_id):
        with self._lock:
            record = self._camera_metrics.get(camera_id)
            if record is None:
                record = self._new_camera_metrics()
            return {
                **record['metrics'],
                'last_scan': record['last_scan'].copy() if record['last_scan'] else None,
                'last_candidate': (
                    record['last_candidate'].copy()
                    if record['last_candidate'] else None
                ),
                'last_notification': (
                    record['last_notification'].copy()
                    if record['last_notification'] else None
                ),
            }

    def submit(self, key, image, cue_threshold, callback):
        cue_threshold = validate_threshold('cue_threshold', cue_threshold)
        if image is None or not getattr(image, 'size', 0):
            raise ValueError('person crop must be a non-empty image')
        if not callable(callback):
            raise ValueError('callback must be callable')
        now = time.monotonic()
        camera_id = key[0] if isinstance(key, tuple) and key else None
        with self._lock:
            if key in self._pending:
                self._increment('skipped_pending', camera_id)
                return False
            last_submitted = self._last_submitted.get(key)
            if (
                last_submitted is not None
                and now - last_submitted < self.scan_cooldown_seconds
            ):
                self._increment('skipped_cooldown', camera_id)
                return False
            job = _ScanJob(key, image.copy(), cue_threshold, callback)
            try:
                self._queue.put_nowait(job)
            except queue.Full:
                self._increment('skipped_queue_full', camera_id)
                logger.warning(
                    'Concealment cue queue is full; skipping scan for key %r',
                    key,
                )
                return False
            self._pending.add(key)
            self._last_submitted[key] = now
            self._increment('submitted', camera_id)
            if len(self._last_submitted) > 4096:
                cutoff = now - max(self.scan_cooldown_seconds * 2, 60.0)
                self._last_submitted = {
                    item_key: submitted_at
                    for item_key, submitted_at in self._last_submitted.items()
                    if submitted_at >= cutoff or item_key in self._pending
                }
            if self._worker is None or not self._worker.is_alive():
                self._worker = threading.Thread(
                    target=self._run,
                    name='concealment-cue-worker',
                    daemon=True,
                )
                self._worker.start()
        return True

    def _run(self):
        try:
            self._worker_priority()
        except Exception:
            logger.exception('Unable to lower concealment worker priority')
        while True:
            job = self._queue.get()
            started_at = time.monotonic()
            with self._lock:
                camera_id = job.key[0] if isinstance(job.key, tuple) and job.key else None
                self._increment('started', camera_id)
                self._last_scan_started_at = datetime.now(timezone.utc).isoformat()
                self._active_scan_started_at = self._last_scan_started_at
            try:
                try:
                    if self._model is None:
                        self._model = self._model_factory(
                            self.model_path,
                            prompts=CUE_PROMPTS,
                        )
                    result = scan_person_crop_with_model(
                        self._model,
                        job.image,
                        job.cue_threshold,
                        image_size=self.image_size,
                    )
                except Exception as exc:
                    logger.exception('Background concealment cue scan failed for key %r', job.key)
                    with self._lock:
                        self._increment('failed', camera_id)
                        scan_record = {
                            'outcome': 'failed',
                            'error': f'{type(exc).__name__}: {exc}',
                            'started_at': self._last_scan_started_at,
                            'finished_at': datetime.now(timezone.utc).isoformat(),
                            'duration_ms': round((time.monotonic() - started_at) * 1000, 1),
                        }
                        self._last_scan = scan_record
                        record = self._camera_record(camera_id)
                        if record is not None:
                            record['last_scan'] = scan_record.copy()
                    self._notify(job, None, exc)
                else:
                    with self._lock:
                        self._increment('completed', camera_id)
                        if result.review_required:
                            self._increment('combined_cues_found', camera_id)
                        else:
                            self._increment('no_combined_cues', camera_id)
                        scan_record = {
                            'outcome': (
                                'combined_cues_found'
                                if result.review_required
                                else 'no_combined_cues'
                            ),
                            'face_covering_score': result.face_covering_score,
                            'headwear_score': result.headwear_score,
                            'threshold': result.cue_threshold,
                            'started_at': self._last_scan_started_at,
                            'finished_at': datetime.now(timezone.utc).isoformat(),
                            'duration_ms': round((time.monotonic() - started_at) * 1000, 1),
                        }
                        self._last_scan = scan_record
                        record = self._camera_record(camera_id)
                        if record is not None:
                            record['last_scan'] = scan_record.copy()
                    self._notify(job, result, None)
            finally:
                with self._lock:
                    self._pending.discard(job.key)
                    self._active_scan_started_at = None
                self._queue.task_done()

    @staticmethod
    def _notify(job, result, error):
        try:
            job.callback(result, error)
        except Exception:
            logger.exception('Concealment scan callback failed for key %r', job.key)


def scan_image(
    image_path,
    *,
    model_path=DEFAULT_MODEL,
    confidence=DEFAULT_IMAGE_CONFIDENCE,
    cue_threshold=DEFAULT_CUE_THRESHOLD,
):
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f'Image does not exist: {image_path}')
    confidence = validate_threshold('confidence', confidence)
    cue_threshold = validate_threshold('cue_threshold', cue_threshold)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Unable to decode image: {image_path}')
    model = load_yoloe_model(model_path, PROMPTS)
    detections = _load_detections(model, image, confidence)
    eligible_detections = [item for item in detections if item.confidence >= confidence]
    assessments = associate_cues_with_people(eligible_detections, cue_threshold)
    annotated = annotate_image(
        image,
        assessments,
        detections=detections,
        confidence_threshold=confidence,
        cue_threshold=cue_threshold,
    )
    return assessments, annotated, detections


def annotate_image(
    image,
    assessments,
    *,
    detections=(),
    confidence_threshold=DEFAULT_IMAGE_CONFIDENCE,
    cue_threshold=DEFAULT_CUE_THRESHOLD,
):
    annotated = image.copy()
    known_classes = FACE_COVERING_CLASSES | HEADWEAR_CLASSES | {'person'}
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.box)
        accepted = (
            detection.class_name in known_classes
            and detection.confidence >= confidence_threshold
            and (
                detection.class_name == 'person'
                or (
                    detection.confidence >= cue_threshold
                    and any(
                        _overlap_ratio(detection.box, assessment.box) >= 0.5
                        for assessment in assessments
                    )
                )
            )
        )
        color = (0, 180, 0) if accepted else (128, 128, 128)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            annotated,
            f'{detection.class_name} {detection.confidence:.2f}',
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color,
            1,
            cv2.LINE_AA,
        )
    for assessment in assessments:
        x1, y1, x2, y2 = assessment.box
        color = (0, 0, 255) if assessment.review_required else (0, 180, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = assessment.category.replace('_', ' ')
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return annotated

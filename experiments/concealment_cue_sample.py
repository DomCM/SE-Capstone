"""Standalone CPU image scan for observable face-covering/headwear cues.

The detector identifies visual cues only. A combined cue is a prompt for
human review and does not imply intent or wrongdoing.
"""

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Optional, Tuple

import cv2


PROMPTS = (
    'person',
    'face mask',
    'surgical mask',
    'cloth face covering',
    'balaclava',
    'baseball cap',
    'cap',
    'hat',
    'beanie',
    "person wearing a face covering",
)
PERSON_CLASSES = {'person'}
FACE_COVERING_CLASSES = {'face mask', 'surgical mask', 'cloth face covering', 'balaclava'}
HEADWEAR_CLASSES = {'baseball cap', 'cap', 'hat', 'beanie'}
DEFAULT_MODEL = 'yoloe-26s-seg.pt'


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
    """Interpret cue scores without inferring a person's intent."""
    if not isinstance(face_assessable, bool):
        raise ValueError('face_assessable must be a boolean')
    if identity_recognized is not None and not isinstance(identity_recognized, bool):
        raise ValueError('identity_recognized must be a boolean or None')
    if isinstance(cue_threshold, bool) or not isinstance(cue_threshold, (int, float)):
        raise ValueError('cue_threshold must be a number between 0 and 1')
    if not math.isfinite(cue_threshold) or cue_threshold < 0 or cue_threshold > 1:
        raise ValueError('cue_threshold must be a finite number between 0 and 1')

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
            review_required=False,
            cues=cues,
            explanation='One appearance cue was detected; this alone is not a combined-cue alert.',
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
    """Return the fraction of the cue box covered by a person box."""
    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box
    intersection_width = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    intersection_height = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inner_area == 0:
        return 0.0
    return intersection_width * intersection_height / inner_area


def associate_cues_with_people(detections, cue_threshold=0.20):
    """Assign accessory detections to person boxes and assess each person."""
    if isinstance(cue_threshold, bool) or not isinstance(cue_threshold, (int, float)):
        raise ValueError('cue_threshold must be a number between 0 and 1')
    if not math.isfinite(cue_threshold) or cue_threshold < 0 or cue_threshold > 1:
        raise ValueError('cue_threshold must be a finite number between 0 and 1')

    people = [item for item in detections if item.class_name in PERSON_CLASSES]
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
        if len(cues) == 2:
            category = 'multiple_cues_for_review'
            review_required = True
            explanation = (
                'Face covering and headwear detections overlap this person. '
                'Human review is required; intent is unknown.'
            )
        elif cues:
            category = 'single_cue_observed'
            review_required = False
            explanation = 'One appearance cue was detected; this alone is not a combined-cue alert.'
        else:
            category = 'no_combined_cues_detected'
            review_required = False
            explanation = (
                'No combined face-covering and headwear cues were detected. '
                'This does not establish whether the face is identifiable.'
            )
        assessments.append(PersonImageAssessment(
            box=tuple(int(round(value)) for value in person.box),
            face_covering_score=face_covering_score,
            headwear_score=headwear_score,
            category=category,
            review_required=review_required,
            cues=cues,
            explanation=explanation,
        ))
    return assessments


def _load_detections(model, image, confidence):
    result = model.predict(
        source=image,
        conf=confidence,
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


def _annotate_image(image, assessments):
    annotated = image.copy()
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


def scan_image(
    image_path,
    *,
    model_path=DEFAULT_MODEL,
    confidence=0.20,
    cue_threshold=0.20,
):
    """Scan one local image on CPU and return person cue assessments + overlay."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f'Image does not exist: {image_path}')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError('confidence must be a number between 0 and 1')
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError('confidence must be a finite number between 0 and 1')
    if isinstance(cue_threshold, bool) or not isinstance(cue_threshold, (int, float)):
        raise ValueError('cue_threshold must be a number between 0 and 1')
    if not math.isfinite(cue_threshold) or cue_threshold < 0 or cue_threshold > 1:
        raise ValueError('cue_threshold must be a finite number between 0 and 1')

    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f'Unable to decode image: {image_path}')

    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise RuntimeError(
            'Ultralytics is required for image scanning; install project requirements first.'
        ) from exc

    model = YOLOE(str(model_path))
    model.set_classes(list(PROMPTS))
    detections = _load_detections(model, image, confidence)
    assessments = associate_cues_with_people(detections, cue_threshold)
    return assessments, _annotate_image(image, assessments)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Scan one image for visible face-covering/headwear cues for human review.'
    )
    parser.add_argument('image', help='Path to a local image file')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        help=f'YOLOE model path or Ultralytics model name (default: {DEFAULT_MODEL})',
    )
    parser.add_argument('--confidence', type=float, default=0.20, help='Minimum detector confidence')
    parser.add_argument('--cue-threshold', type=float, default=0.20, help='Minimum cue detection confidence')
    parser.add_argument('--output', help='Annotated image output path')
    args = parser.parse_args(argv)

    assessments, annotated = scan_image(
        args.image,
        model_path=args.model,
        confidence=args.confidence,
        cue_threshold=args.cue_threshold,
    )
    image_path = Path(args.image)
    output_path = Path(args.output) if args.output else image_path.with_name(
        f'{image_path.stem}_cue_scan.jpg'
    )
    if output_path.resolve() == image_path.resolve():
        parser.error('Output image path must not overwrite the input image')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f'Unable to write annotated image: {output_path}')

    print(json.dumps({
        'image': str(image_path),
        'annotated_image': str(output_path),
        'model': args.model,
        'device': 'cpu',
        'confidence_threshold': args.confidence,
        'cue_threshold': args.cue_threshold,
        'people_detected': len(assessments),
        'review_required': any(item.review_required for item in assessments),
        'assessments': [asdict(item) for item in assessments],
        'notice': 'Visual cues only. Review detections manually; no intent is inferred.',
    }, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

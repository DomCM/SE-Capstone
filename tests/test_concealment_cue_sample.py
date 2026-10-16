import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from experiments.concealment_cue_sample import (
    Detection,
    assess_person_cues,
    associate_cues_with_people,
)


def test_face_covering_and_headwear_are_a_review_cue_only():
    result = assess_person_cues(
        face_assessable=False,
        face_covering_score=0.91,
        headwear_score=0.82,
    )

    assert result.category == 'multiple_cues_for_review'
    assert result.review_required is True
    assert result.cues == ('face_covering', 'headwear')
    assert 'intent is unknown' in result.explanation


def test_unrecognized_but_assessable_face_without_cues_is_not_flagged():
    result = assess_person_cues(
        face_assessable=True,
        face_covering_score=0.20,
        headwear_score=0.30,
        identity_recognized=False,
    )

    assert result.category == 'unrecognized_face_no_cues'
    assert result.review_required is False


def test_unassessable_face_without_positive_cues_is_ambiguous():
    result = assess_person_cues(
        face_assessable=False,
        face_covering_score=0.25,
        headwear_score=None,
    )

    assert result.category == 'face_not_assessable'
    assert result.review_required is False
    assert result.cues == ()


def test_one_positive_cue_does_not_trigger_combined_review_flag():
    result = assess_person_cues(
        face_assessable=True,
        face_covering_score=0.90,
        headwear_score=0.69,
        identity_recognized=False,
    )

    assert result.category == 'single_cue_observed'
    assert result.review_required is False
    assert result.cues == ('face_covering',)


@pytest.mark.parametrize('score', [-0.1, 1.1, float('nan'), float('inf'), True])
def test_invalid_confidence_scores_are_rejected(score):
    with pytest.raises(ValueError):
        assess_person_cues(
            face_assessable=True,
            face_covering_score=score,
            headwear_score=None,
        )


def test_combined_cues_are_associated_with_the_same_person():
    detections = [
        Detection('person', 0.95, (10, 10, 110, 210)),
        Detection('face mask', 0.80, (40, 50, 80, 100)),
        Detection('cap', 0.75, (35, 15, 85, 60)),
    ]

    results = associate_cues_with_people(detections)

    assert len(results) == 1
    assert results[0].review_required is True
    assert results[0].cues == ('face_covering', 'headwear')
    assert results[0].category == 'multiple_cues_for_review'


def test_cues_on_separate_people_do_not_trigger_combined_review():
    detections = [
        Detection('person', 0.95, (0, 0, 100, 200)),
        Detection('person', 0.94, (200, 0, 300, 200)),
        Detection('face mask', 0.80, (30, 30, 70, 80)),
        Detection('cap', 0.75, (230, 5, 270, 45)),
    ]

    results = associate_cues_with_people(detections)

    assert len(results) == 2
    assert [result.review_required for result in results] == [False, False]
    assert [result.cues for result in results] == [
        ('face_covering',),
        ('headwear',),
    ]


def test_scanner_rejects_missing_image_before_loading_model(tmp_path):
    from experiments.concealment_cue_sample import scan_image

    with pytest.raises(FileNotFoundError, match='Image does not exist'):
        scan_image(tmp_path / 'missing.jpg')


def test_scan_image_uses_cpu_model_and_returns_assessment_and_overlay(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    import ultralytics

    from experiments.concealment_cue_sample import PROMPTS, scan_image

    image_path = tmp_path / 'sample.jpg'
    assert cv2.imwrite(str(image_path), np.zeros((120, 120, 3), dtype=np.uint8))

    class FakeBox:
        def __init__(self, class_id, confidence, coordinates):
            self.cls = [FakeScalar(class_id)]
            self.conf = [FakeScalar(confidence)]
            self.xyxy = [FakeCoordinates(coordinates)]

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

    calls = {}

    class FakeModel:
        def set_classes(self, classes):
            calls['classes'] = classes

        def predict(self, **kwargs):
            calls['predict'] = kwargs
            return [type('Result', (), {
                'names': {0: 'person', 1: 'face mask', 2: 'cap'},
                'boxes': [
                    FakeBox(0, 0.95, [10, 10, 110, 115]),
                    FakeBox(1, 0.85, [35, 35, 75, 65]),
                    FakeBox(2, 0.80, [30, 12, 80, 40]),
                ],
            })()]

    def create_model(model_path):
        calls['model_path'] = model_path
        return FakeModel()

    monkeypatch.setattr(ultralytics, 'YOLOE', create_model)

    assessments, overlay = scan_image(image_path)

    assert calls['model_path'] == 'yoloe-26s-seg.pt'
    assert calls['classes'] == list(PROMPTS)
    assert calls['predict']['device'] == 'cpu'
    assert len(assessments) == 1
    assert assessments[0].review_required is True
    assert overlay.shape == (120, 120, 3)

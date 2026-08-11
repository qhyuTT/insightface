from __future__ import annotations

import numpy as np

from person_search.detector import _decode_yolox, _nms, _preprocess


def test_preprocess_letterboxes_to_model_size() -> None:
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    result, ratio = _preprocess(image, (416, 416))
    assert result.shape == (3, 416, 416)
    assert ratio == 2.08


def test_decode_yolox_builds_all_three_feature_grids() -> None:
    prediction_count = 52 * 52 + 26 * 26 + 13 * 13
    raw = np.zeros((prediction_count, 85), dtype=np.float32)
    decoded = _decode_yolox(raw, (416, 416))
    assert decoded.shape == raw.shape
    np.testing.assert_allclose(decoded[0, :4], [0, 0, 8, 8])


def test_nms_suppresses_overlapping_lower_score_box() -> None:
    boxes = np.asarray([[0, 0, 100, 100], [5, 5, 95, 95], [200, 200, 250, 250]])
    scores = np.asarray([0.9, 0.8, 0.7])
    assert _nms(boxes, scores, 0.5) == [0, 2]

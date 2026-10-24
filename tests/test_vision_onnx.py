"""Exercise the ONNX Runtime code path with tiny fake models that have the
same inputs/outputs as the real mmdeploy exports (no downloads needed)."""

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper, numpy_helper  # noqa: E402

from motionplay.types import Frame  # noqa: E402
from motionplay.vision.models import ModelSpec, parse_pipeline  # noqa: E402
from motionplay.vision.rtm import (PersonDetector, RTMDetPoseBackend,  # noqa: E402
                                   TopDownPose, crop_transform)


def _save(graph, path):
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _zero_from_input(name="input"):
    """Nodes producing zero shaped (B,1,1) from the input, so outputs carry batch size."""
    return [
        helper.make_node("ReduceMean", [name], ["m"], axes=[1, 2, 3], keepdims=1),
        helper.make_node("Reshape", ["m", "shape_b11"], ["m3"]),
        helper.make_node("Mul", ["m3", "zero"], ["z"]),
    ]


def make_fake_pose(path, peaks_xy, k=26, w=192, h=256, split=2.0):
    sx = np.full((1, k, int(w * split)), -0.1, np.float32)
    sy = np.full((1, k, int(h * split)), -0.1, np.float32)
    for j, (x, y) in enumerate(peaks_xy):
        sx[0, j, int(x * split)] = 0.9
        sy[0, j, int(y * split)] = 0.7
    nodes = _zero_from_input() + [
        helper.make_node("Add", ["z", "cx"], ["simcc_x"]),
        helper.make_node("Add", ["z", "cy"], ["simcc_y"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_rtmpose",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3, h, w])],
        [helper.make_tensor_value_info("simcc_x", TensorProto.FLOAT, ["batch", k, int(w * split)]),
         helper.make_tensor_value_info("simcc_y", TensorProto.FLOAT, ["batch", k, int(h * split)])],
        [numpy_helper.from_array(np.array([-1, 1, 1], np.int64), "shape_b11"),
         numpy_helper.from_array(np.zeros((1,), np.float32), "zero"),
         numpy_helper.from_array(sx, "cx"), numpy_helper.from_array(sy, "cy")])
    _save(graph, path)


def make_fake_detector(path, dets, labels, size=320):
    nodes = _zero_from_input() + [
        helper.make_node("Add", ["z", "d"], ["dets"]),
        helper.make_node("Identity", ["l"], ["labels"]),
    ]
    graph = helper.make_graph(
        nodes, "fake_rtmdet",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, size, size])],
        [helper.make_tensor_value_info("dets", TensorProto.FLOAT, [1, len(dets), 5]),
         helper.make_tensor_value_info("labels", TensorProto.INT64, [1, len(dets)])],
        [numpy_helper.from_array(np.array([-1, 1, 1], np.int64), "shape_b11"),
         numpy_helper.from_array(np.zeros((1,), np.float32), "zero"),
         numpy_helper.from_array(np.asarray(dets, np.float32)[None], "d"),
         numpy_helper.from_array(np.asarray(labels, np.int64)[None], "l")])
    _save(graph, path)


POSE_SPEC = ModelSpec("fake-pose", "pose_topdown", "x", input_size=(192, 256),
                      mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375), to_rgb=True)
DET_SPEC = ModelSpec("fake-det", "detector", "x", input_size=(320, 320),
                     mean=(103.53, 116.28, 123.675), std=(57.375, 57.12, 58.395))


def test_topdown_pose_maps_crop_back_to_image(tmp_path):
    # Every keypoint peaks at the crop centre, joint 0 at the crop top-left quarter.
    peaks = [(96, 128)] * 26
    peaks[0] = (48, 64)
    make_fake_pose(tmp_path / "pose.onnx", peaks)
    pose = TopDownPose(tmp_path / "pose.onnx", POSE_SPEC, ["cpu"], fp16=False, max_batch=4)
    assert pose.batched
    image = np.zeros((720, 1280, 3), np.uint8)
    boxes = np.array([[100, 50, 300, 450], [700, 100, 900, 500], [400, 200, 500, 400]], np.float32)
    kpts, scores = pose(image, boxes)
    assert kpts.shape == (3, 26, 2)
    for box, kp in zip(boxes, kpts):
        center, scale, _ = crop_transform(box, (192, 256))
        np.testing.assert_allclose(kp[5], center, atol=1.0)
        np.testing.assert_allclose(kp[0], center - scale / 4, atol=1.0)
    np.testing.assert_allclose(scores, 0.8, atol=1e-5)


def test_detector_keeps_only_people_and_unscales(tmp_path):
    # Letterbox: 1280x720 -> ratio 0.25 into 320x320.
    dets = [[25, 10, 75, 160, 0.9], [100, 20, 150, 100, 0.8], [200, 50, 210, 60, 0.05]]
    make_fake_detector(tmp_path / "det.onnx", dets, [0, 1, 0])
    det = PersonDetector(tmp_path / "det.onnx", DET_SPEC, ["cpu"], fp16=False)
    out = det(np.zeros((720, 1280, 3), np.uint8), score_thr=0.15)
    assert out.shape == (1, 5)
    np.testing.assert_allclose(out[0, :4], [100, 40, 300, 640])


def test_backend_glue(tmp_path):
    make_fake_pose(tmp_path / "pose.onnx", [(96, 128)] * 26)
    make_fake_detector(tmp_path / "det.onnx", [[25, 10, 75, 160, 0.9]], [0])
    backend = RTMDetPoseBackend(
        PersonDetector(tmp_path / "det.onnx", DET_SPEC, ["cpu"], False),
        TopDownPose(tmp_path / "pose.onnx", POSE_SPEC, ["cpu"], False), "fake")
    frame = Frame(np.zeros((720, 1280, 3), np.uint8), 0.0, 1)
    boxes = backend.detect(frame)
    cands = backend.estimate(frame, boxes)
    assert len(cands) == 1 and cands[0].det_score == pytest.approx(0.9)


def test_parse_mmdeploy_pipeline():
    data = {"pipeline": {"tasks": [
        {"transforms": [{"type": "TopDownAffine", "input_size": [192, 256]},
                        {"type": "Normalize", "mean": [1, 2, 3], "std": [4, 5, 6], "to_rgb": True}]},
        {"name": "postprocess", "params": {"simcc_split_ratio": 2.0}},
    ]}}
    found = parse_pipeline(data)
    assert found["input_size"] == (192, 256)
    assert found["mean"] == (1.0, 2.0, 3.0) and found["to_rgb"] is True
    assert found["extra"]["simcc_split_ratio"] == 2.0

"""Graphe ONNX élagué (DECISIONS n° 180) et session OpenVINO à la place d'ONNX Runtime
(processeur graphique intégré, n° 179)."""

import numpy as np
import pytest

from blanci.embedding.encoders.openvino_session import openvino_session
from blanci.embedding.encoders.pruned import PrunedSession, pruned_model

onnx = pytest.importorskip("onnx")


@pytest.fixture
def model(tmp_path):
    """Graphe jouet : (lot, 8) → le double, et la somme de chaque ligne."""
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "x"], ["double"]),
            helper.make_node("ReduceSum", ["x", "axes"], ["total"], keepdims=0),
        ],
        "jouet",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 8])],
        [
            helper.make_tensor_value_info("double", TensorProto.FLOAT, ["batch", 8]),
            helper.make_tensor_value_info("total", TensorProto.FLOAT, ["batch"]),
        ],
        [helper.make_tensor("axes", TensorProto.INT64, [1], [1])],
    )
    path = tmp_path / "jouet.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]), path)
    return path


@pytest.mark.parametrize("n", [1, 4, 7])  # lot plus court, égal, plus long que celui du graphe
def test_any_batch_goes_through_the_fixed_size_graph(model, tmp_path, n):
    pytest.importorskip("openvino")
    session = openvino_session(model, {"device": "CPU", "batch_size": 4}, tmp_path / "cache")
    x = np.random.default_rng(0).normal(size=(n, 8)).astype(np.float32)
    double, total = session.run(None, {"x": x})
    np.testing.assert_allclose(double, 2 * x, rtol=1e-6)
    np.testing.assert_allclose(total, x.sum(axis=1), rtol=1e-5)


def test_missing_device_falls_back_to_none(model, capsys):
    pytest.importorskip("openvino")
    assert openvino_session(model, {"device": "ABSENT"}) is None
    assert "ONNX Runtime" in capsys.readouterr().out


def test_pruned_graph_keeps_its_outputs_and_fills_the_removed_ones(model, tmp_path):
    import onnxruntime as ort

    small = pruned_model(model, ["total"], tmp_path / "cache")
    assert small != model and pruned_model(model, ["total"], tmp_path / "cache") == small
    assert [n.op_type for n in onnx.load(small).graph.node] == ["ReduceSum"]  # « Add » retiré
    session = PrunedSession(ort.InferenceSession(str(small)), ["total"], ["double", "total"])
    x = np.random.default_rng(1).normal(size=(3, 8)).astype(np.float32)
    double, total = session.run(None, {"x": x})
    assert double.shape == (3, 0)
    full = ort.InferenceSession(str(model)).run(["total"], {"x": x})[0]
    np.testing.assert_array_equal(total, full)

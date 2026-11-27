import json

import numpy as np
import pytest


def test_frozen_split_changes_only_selected_training_inputs():
    from examples.MoERobustness.data import make_cohort

    labels = np.repeat(np.arange(10, dtype=np.uint8), 20)
    pixels = np.broadcast_to(labels[:, None, None] * 20, (200, 28, 28)).copy()
    first = make_cohort(pixels, labels, train_per_digit=10, val_per_digit=5)
    second = make_cohort(pixels, labels, train_per_digit=10, val_per_digit=5)
    assert len(first.train_ids) == 100 and len(first.val_ids) == 50
    assert not set(first.train_ids).intersection(first.val_ids)
    np.testing.assert_array_equal(first.train_ids, second.train_ids)
    np.testing.assert_array_equal(first.pairs, second.pairs)
    np.testing.assert_array_equal(first.labels, labels[np.r_[first.train_ids, first.val_ids]])
    changed = np.flatnonzero(np.any(first.clean != first.proxy, axis=(1, 2)))
    assert len(changed) == 3
    assert set(first.train_ids[changed]) == set(first.pairs[:, 0])
    assert (first.labels[changed] == 9).all()
    assert (first.proxy[changed] == 90).all()
    assert set(first.pairs[:, 1]).issubset(first.train_ids)


def test_idx_parser_rejects_bad_length_and_magic():
    from examples.MoERobustness.data import decode_idx

    data = bytes.fromhex("0000080100000003") + bytes([0, 1, 9])
    np.testing.assert_array_equal(decode_idx(data, 1), [0, 1, 9])
    with pytest.raises(ValueError, match="size"):
        decode_idx(data[:-1], 1)
    with pytest.raises(ValueError, match="header"):
        decode_idx(data, 3)


def test_shared_weights_and_router_gradient():
    import torch
    from examples.MoERobustness.models import DenseDigitClassifier, SparseDigitClassifier

    torch.manual_seed(11)
    dense = DenseDigitClassifier(dataset=None)
    torch.manual_seed(11)
    sparse = SparseDigitClassifier(dataset=None)
    for key, tensor in dense.state_dict().items():
        if key.startswith(("features.", "head.")):
            torch.testing.assert_close(tensor, sparse.state_dict()[key], rtol=0, atol=0)
    images = torch.rand(16, 1, 28, 28)
    prediction = sparse.predict(images)
    assert prediction.logits.shape == (16, 10)
    assert prediction.expert.shape == (16,)
    assert prediction.expert.min() >= 0 and prediction.expert.max() < 4
    torch.nn.functional.cross_entropy(prediction.logits, torch.arange(16) % 10).backward()
    assert torch.count_nonzero(sparse.router.weight.grad) > 0


def test_top1_executes_only_selected_expert():
    import torch
    from examples.MoERobustness.models import SparseDigitClassifier

    model = SparseDigitClassifier(dataset=None)
    with torch.no_grad():
        model.router.weight.zero_()
        model.router.bias.copy_(torch.tensor([0., 1., 100., -1.]))
    calls = []
    for index, expert in enumerate(model.experts):
        expert.register_forward_hook(lambda *args, slot=index: calls.append(slot))
    prediction = model.predict(torch.zeros(5, 1, 28, 28))
    assert calls == [2]
    assert prediction.expert.tolist() == [2] * 5
    assert torch.isfinite(prediction.logits).all()


def test_confusion_and_collateral_f1():
    from examples.MoERobustness.metrics import class_metrics

    summary = class_metrics(np.array([4, 4, 4, 9]), np.array([4, 4, 9, 4]))
    assert summary["confusion"][4][9] == 1
    assert summary["confusion"][9][4] == 1
    assert summary["f1"][4] == pytest.approx(2 / 3)
    assert summary["bystander_macro_f1"] == 0


def test_smoke_runs_normal_entrypoint_and_preserves_outputs(tmp_path):
    from examples.MoERobustness.run import main

    out = tmp_path / "attempt"
    argv = ["--mode", "smoke", "--arch", "moe", "--data", "proxy", "--seed", "11",
            "--dataset-dir", str(tmp_path / "absent"), "--output-dir", str(out)]
    assert main(argv) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    metrics = json.loads((out / "summary.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["mode"] == "smoke"
    assert metrics["train"]["examples"] == 100
    assert metrics["validation"]["examples"] == 50
    assert metrics["test"]["examples"] == 80
    assert len(metrics["diagnostic"]["assignment_counts"]) == 4
    assert len((out / "metrics.jsonl").read_text().splitlines()) == 1
    with np.load(out / "predictions.npz", allow_pickle=False) as predictions:
        assert predictions["logits"].shape == (80, 10)
        assert predictions["expert"].shape == (80,)
    before = (out / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        main(argv)
    assert (out / "manifest.json").read_bytes() == before


def test_pair_delta_preserves_sign_and_requires_four_arms():
    from examples.MoERobustness.summarize import paired_class4

    f1 = {("dense", "clean"): 0.97, ("dense", "proxy"): 0.87,
          ("moe", "clean"): 0.95, ("moe", "proxy"): 0.9}
    assert paired_class4(f1) == pytest.approx(0.05)
    f1.pop(("moe", "proxy"))
    with pytest.raises(ValueError, match="four"):
        paired_class4(f1)

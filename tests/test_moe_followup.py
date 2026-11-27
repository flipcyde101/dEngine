"""Regression checks for the follow-up experiment's scientific controls."""

import numpy as np
import pytest

from examples.MoERobustness.followup.cohort import eligible_images, proxy_pixels
from examples.MoERobustness.followup.design import ARMS, arm_order, interpretation


def test_reserving_images_excludes_prior_examples_and_pixel_duplicates():
    images = np.array([[7], [7], [8], [9], [9], [10]], dtype=np.uint8)
    old_test = np.array([[10]], dtype=np.uint8)
    assert eligible_images(images, {1}, old_test).tolist() == [2, 3]


def test_corruption_changes_selected_training_9_only_and_retains_labels():
    images = np.broadcast_to(np.array([0, 100, 24], dtype=np.uint8)[:, None, None],
                             (3, 28, 28)).copy()
    labels = np.array([9, 4, 9], dtype=np.uint8)
    ids = [25, 42, 51]
    proxy = proxy_pixels(images, labels, ids, [(25, 42)], 2)
    assert np.all(proxy[0] == 90)
    np.testing.assert_array_equal(proxy[1:], images[1:])
    assert np.all(images[0] == 0)
    assert labels.tolist() == [9, 4, 9]
    with pytest.raises(ValueError, match="training"):
        proxy_pixels(images, labels, ids, [(51, 42)], 2)
    with pytest.raises(ValueError, match="9.*4"):
        proxy_pixels(images, labels, ids, [(42, 25)], 2)
    with pytest.raises(ValueError, match="28"):
        proxy_pixels(images[:, :27], labels, ids, [(25, 42)], 2)


def test_arm_order_balances_every_position_over_six_starts():
    orders = [arm_order(i) for i in range(6)]
    assert all(set(order) == set(ARMS) for order in orders)
    assert all(set(position) == set(ARMS) for position in zip(*orders))


@pytest.mark.parametrize(("low", "high", "expected"), [
    (2.01, 4, "material_advantage"),
    (-4, -2.01, "material_disadvantage"),
    (-1.8, 1.9, "small_average_difference"),
    (-1, 2.1, "unresolved"),
    (2, 3, "unresolved"),
])
def test_practical_conclusion_requires_the_whole_interval(low, high, expected):
    assert interpretation(low, high, 2) == expected


def test_capacity_control_preserves_shared_initialization():
    import torch
    from examples.MoERobustness.followup.fit import ARCHITECTURES, parameter_count, weights_digest
    from examples.MoERobustness.run import tensor_dataset

    data = tensor_dataset(np.zeros((8, 28, 28), dtype=np.uint8), np.arange(8, dtype=np.uint8))
    shared = set()
    counts = {}
    for key, model_class in ARCHITECTURES.items():
        torch.manual_seed(812)
        model = model_class(data)
        counts[key] = parameter_count(model)
        shared.add(weights_digest({name: value for name, value in model.state_dict().items()
                                   if name.startswith(("features.", "head."))}))
    assert counts == {"dense": 60458, "wide": 85742, "moe": 85678}
    assert len(shared) == 1


def test_recorded_sampler_measures_the_order_consumed():
    from examples.MoERobustness.followup.fit import RecordedSampler

    recorded = RecordedSampler([2, 0, 1])
    assert list(recorded) == [2, 0, 1]
    assert recorded.epochs[0]["examples"] == 3
    other = RecordedSampler([0, 2, 1])
    list(other)
    assert recorded.epochs[0]["order_sha256"] != other.epochs[0]["order_sha256"]


def test_development_factory_declares_the_type_required_by_dengine():
    from dengine.dataset import SupervisedDataset
    from examples.MoERobustness.followup.fit import development_data

    assert development_data.__annotations__["return"] is SupervisedDataset


def test_rejects_hidden_mismatch_in_matched_conditions():
    from examples.MoERobustness.followup.verify import verify_pairing

    records = {arm: {"initial_shared": "same", "initial_model": arm.split("-")[0],
                     "batch_order": [0, 1]} for arm in ARMS}
    verify_pairing(records)
    del records["dense-proxy"]
    with pytest.raises(ValueError, match="All six"):
        verify_pairing(records)
    records["dense-proxy"] = {"initial_shared": "same", "initial_model": "dense", "batch_order": [1, 0]}
    with pytest.raises(ValueError, match="batch order"):
        verify_pairing(records)
    records["dense-proxy"]["batch_order"] = [0, 1]
    records["dense-proxy"]["initial_model"] = "different"
    with pytest.raises(ValueError, match="initial weights"):
        verify_pairing(records)


def test_pair_loss_sign_and_interval_use_cohort_observations():
    from examples.MoERobustness.followup.score import paired_loss, interval

    stats = {name: {"f1": [0] * 4 + [f1]} for name, f1 in {
        "dense-clean": .98, "dense-proxy": .91, "moe-clean": .96, "moe-proxy": .94,
    }.items()}
    assert paired_loss(stats, "dense", "moe") == pytest.approx(5)
    result = interval([0, 2, 4])
    assert result["mean_pp"] == 2
    assert result["low_pp"] < 0 < 4 < result["high_pp"]
    assert result["df"] == 2


def test_provider_must_execute_the_reviewed_arguments():
    from copy import deepcopy
    from examples.MoERobustness.followup.score import verify_provider

    configuration = {"serviceAccount": "restricted", "timeout": "600s", "queueTtl": "300s",
                     "tags": ["one"], "options": {"machineType": "E2_STANDARD_2"},
                     "steps": [{"id": "step", "name": "image", "entrypoint": "sh", "args": ["$$HOME"]}]}
    observed = deepcopy(configuration)
    observed["steps"][0]["args"] = ["$HOME"]
    verify_provider(configuration, observed)
    observed["steps"][0]["args"] = ["$PATH"]
    with pytest.raises(ValueError, match="arguments"):
        verify_provider(configuration, observed)


@pytest.fixture
def matched_class_scores():
    rows = []
    for cohort in range(1, 21):
        for slot in range(1, 5):
            scores = {arm: {"f1": [.95] * 10} for arm in ARMS}
            scores["dense-proxy"]["f1"][0] = .94
            scores["moe-proxy"]["f1"][0] = .91
            scores["dense-proxy"]["f1"][4] = .90 - cohort / 1000 - slot / 10000
            scores["moe-proxy"]["f1"][4] = .93
            scores["wide-proxy"]["f1"][0] = .90
            scores["wide-proxy"]["f1"][4] = .88 - cohort / 1000 - slot / 10000
            rows.append({"cohort": cohort, "slot": slot, "scores": scores})
    return rows


def test_class_report_uses_cohort_means_and_the_dense_minus_expert_loss(matched_class_scores):
    from examples.MoERobustness.followup.class_report import summarize_classes

    result = summarize_classes(list(reversed(matched_class_scores)))
    assert result["matched_starts"] == 80
    assert result["cohorts"] == 20
    assert result["starts_per_cohort"] == 4
    assert result["scope"] == "secondary_exploratory"
    assert result["interval_adjustment"] == "none"
    assert [row["digit"] for row in result["classes"]] == list(range(10))
    assert result["classes"][0]["summary"]["mean_pp"] == pytest.approx(-3)
    digit_four = result["classes"][4]
    assert digit_four["cohort_means_pp"][0] == pytest.approx(3.125)
    assert digit_four["cohort_means_pp"][-1] == pytest.approx(5.025)
    assert digit_four["summary"]["mean_pp"] == pytest.approx(4.075)
    assert digit_four["summary"]["df"] == 19
    # Independent t table value and hand-derived standard error for 20 equally spaced means.
    half = 2.093024054408263 * (.35 / 20) ** .5
    assert digit_four["summary"]["low_pp"] == pytest.approx(4.075 - half)
    assert digit_four["summary"]["high_pp"] == pytest.approx(4.075 + half)
    assert result["classes"][1]["summary"]["low_pp"] == pytest.approx(0)


def test_class_report_capacity_control_uses_wide_minus_expert_loss(matched_class_scores):
    from examples.MoERobustness.followup.class_report import summarize_classes

    result = summarize_classes(matched_class_scores)
    assert result["classes"][0]["capacity_control"]["summary"]["mean_pp"] == pytest.approx(1)
    control = result["classes"][4]["capacity_control"]
    assert control["cohort_means_pp"][0] == pytest.approx(5.125)
    assert control["cohort_means_pp"][-1] == pytest.approx(7.025)
    assert control["summary"]["mean_pp"] == pytest.approx(6.075)
    assert control["summary"]["df"] == 19
    half = 2.093024054408263 * (.35 / 20) ** .5
    assert control["summary"]["low_pp"] == pytest.approx(6.075 - half)
    assert control["summary"]["high_pp"] == pytest.approx(6.075 + half)


def test_class_report_interval_accepts_two_cohorts_and_rejects_nonfinite_means():
    from examples.MoERobustness.followup.class_report import interval

    result = interval([0, 2])
    assert result["mean_pp"] == 1
    assert result["df"] == 1
    assert result["low_pp"] == pytest.approx(-11.706204736432095)
    assert result["high_pp"] == pytest.approx(13.706204736432095)
    with pytest.raises(ValueError, match="finite"):
        interval([0, float("nan")])


@pytest.mark.parametrize("broken", [
    "missing", "duplicate", "arm", "wide-arm", "length", "nan", "range", "cohort", "slot", "boolean",
    "boolean-cohort", "boolean-slot", "nonobject",
])
def test_class_report_rejects_incomplete_or_invalid_scores(matched_class_scores, broken):
    from examples.MoERobustness.followup.class_report import summarize_classes

    if broken == "missing":
        matched_class_scores.pop()
    elif broken == "duplicate":
        matched_class_scores[-1] = matched_class_scores[0]
    elif broken == "arm":
        del matched_class_scores[0]["scores"]["moe-proxy"]
    elif broken == "wide-arm":
        del matched_class_scores[0]["scores"]["wide-proxy"]
    elif broken == "length":
        matched_class_scores[0]["scores"]["dense-clean"]["f1"].pop()
    elif broken == "nan":
        matched_class_scores[0]["scores"]["dense-clean"]["f1"][0] = float("nan")
    elif broken == "cohort":
        matched_class_scores[0]["cohort"] = 21
    elif broken == "slot":
        matched_class_scores[0]["slot"] = "1"
    elif broken == "boolean":
        matched_class_scores[0]["scores"]["dense-clean"]["f1"][0] = True
    elif broken == "boolean-cohort":
        matched_class_scores[0]["cohort"] = True
    elif broken == "boolean-slot":
        matched_class_scores[0]["slot"] = True
    elif broken == "nonobject":
        matched_class_scores[0] = "invalid"
    else:
        matched_class_scores[0]["scores"]["dense-clean"]["f1"][0] = 1.1
    with pytest.raises(ValueError):
        summarize_classes(matched_class_scores)


def test_saved_class_report_has_provenance_and_readable_unadjusted_intervals(matched_class_scores, tmp_path, monkeypatch):
    import json
    from hashlib import sha256
    from examples.MoERobustness.followup.class_report import main

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "study-results.json"
    source.write_text(json.dumps({"matched_comparisons": matched_class_scores}), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "class-report"
    assert main([str(source), str(output)]) == 0
    assert source.read_bytes() == original
    summary = json.loads((output / "class-comparisons.json").read_text(encoding="utf-8"))
    assert summary["input_results_sha256"] == sha256(original).hexdigest()
    assert summary["classes"][4]["summary"]["mean_pp"] == pytest.approx(4.075)
    report = (output / "class-comparisons.md").read_text(encoding="utf-8")
    assert "secondary" in report.lower() and "exploratory" in report.lower()
    assert "unadjusted" in report.lower()
    assert "Positive" in report and "expert" in report
    assert "20 cohort means" in report
    assert "| 4 | +4.075 |" in report
    assert "capacity control" in report.lower()
    assert "| 4 | +6.075 |" in report
    with pytest.raises(FileExistsError):
        main([str(source), str(output)])


def test_invalid_saved_scores_are_rejected_before_creating_output(matched_class_scores, tmp_path, monkeypatch):
    import json
    from examples.MoERobustness.followup.class_report import main

    monkeypatch.chdir(tmp_path)
    source = tmp_path / "incomplete.json"
    source.write_text(json.dumps({"matched_comparisons": matched_class_scores[:-1]}), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="complete"):
        main([str(source), str(output)])
    assert not output.exists()

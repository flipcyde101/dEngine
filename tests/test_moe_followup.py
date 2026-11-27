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

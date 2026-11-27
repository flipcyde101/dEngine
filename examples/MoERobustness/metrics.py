"""Count and preserve held-out predictions and route assignments."""

import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from _dengine_cli_paths import write_output_text

from dengine.callbacks.callback import PeriodicCallback
from dengine.callbacks.decorators import register_callback
from dengine.training_strategies.decorators import register_local_training
from dengine.training_strategies.local_update_strategy import (
    EngineWithoutEarlyStopping, training_step_output,
)

from .data import SPEC
from .models import balance_loss


def write_json(path, record, *, exclusive=False):
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    write_output_text(path, text, exclusive=exclusive)


def class_metrics(targets, predicted):
    if (targets.ndim != 1 or targets.shape != predicted.shape or not targets.size
            or not ((targets >= 0) & (targets < 10)).all()
            or not ((predicted >= 0) & (predicted < 10)).all()):
        raise ValueError("Expected matching, valid digit predictions and labels")
    counts = np.bincount(targets.astype(np.int64) * 10 + predicted, minlength=100).reshape(10, 10)
    true_pos = counts.diagonal()
    actual = counts.sum(axis=1)
    guessed = counts.sum(axis=0)

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(10), where=denominator != 0)

    f1 = ratio(2 * true_pos, actual + guessed)
    return {
        "examples": int(counts.sum()), "confusion": counts.tolist(),
        "precision": ratio(true_pos, guessed).tolist(), "recall": ratio(true_pos, actual).tolist(),
        "f1": f1.tolist(), "macro_f1": float(f1.mean()),
        "bystander_macro_f1": float(f1[[0, 1, 2, 3, 5, 6, 7, 8]].mean()),
        "accuracy": float(true_pos.sum() / counts.sum()),
    }


def mean_cross_entropy(targets, logits):
    work = logits.astype(np.float64)
    return float(np.mean(np.logaddexp.reduce(work, axis=1)
                         - work[np.arange(len(targets)), targets]))


@torch.inference_mode()
def evaluate(model, dataset):
    was_training = model.training
    model.eval()
    try:
        logits, labels, routes = [], [], []
        for images, targets in DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0):
            prediction = model.predict(images)
            logits.append(prediction.logits)
            labels.append(targets)
            if prediction.expert is not None:
                routes.append(prediction.expert)
        y = torch.cat(labels).numpy()
        logit = torch.cat(logits).numpy()
        if not np.isfinite(logit).all():
            raise ValueError("Non-finite evaluation logits")
        route = torch.cat(routes).numpy() if routes else np.full(y.shape, -1, dtype=np.int64)
        return y, logit, route
    finally:
        model.train(was_training)


def routing_counts(routes):
    if not routes.size or np.all(routes == -1):
        return []
    return np.bincount(routes, minlength=SPEC["experts"]).tolist()


@register_local_training()
class PilotTrainingEngine(EngineWithoutEarlyStopping):
    def __init__(self, *args, seed, validation_data, **kwargs):
        generator = torch.Generator().manual_seed(seed + 9127)
        super().__init__(*args, validation_data=validation_data,
                         training_DataLoader_kwargs={"generator": generator, "num_workers": 0},
                         **kwargs)
        self.validation_data = validation_data

    def training_step(self, net, optimizer, images, labels):
        net.train()
        optimizer.zero_grad()
        pred = net.predict(images)
        loss = F.cross_entropy(pred.logits, labels)
        if pred.expert is not None:
            loss = loss + SPEC["load_balance_weight"] * balance_loss(pred)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training objective")
        loss.backward()
        optimizer.step()
        return training_step_output(pred.logits.detach(), float(loss.detach()))

    def compute_loss_in_production(self, model, data):
        labels, logits, routes = evaluate(model, data)
        return {"labels": labels, "logits": logits, "routes": routes}


@register_callback()
class PilotMetrics(PeriodicCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out = Path(self.experiment_configuration.experiments_directory_root)
        self.progress = None
        self.validation = None
        self.test = None
        self.predictions = None
        self.started = None
        self.training_examples = None

    def training_epoch_start(self, epoch):
        if epoch == 0:
            self.started = time.monotonic()

    def training_epoch_end(self, epoch, **kwargs):
        trainer = self.client.local_training_engine
        y, logits, routes = evaluate(self.client.model, trainer.validation_data)
        self.validation = class_metrics(y, logits.argmax(axis=1))
        self.validation["cross_entropy"] = mean_cross_entropy(y, logits)
        self.validation["route_counts"] = {
            "all": routing_counts(routes), "digit_4": routing_counts(routes[y == 4]),
            "digit_9": routing_counts(routes[y == 9]),
        }
        elapsed = time.monotonic() - self.started
        with (self.out / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"epoch": epoch + 1, "elapsed_training_s": elapsed,
                                  "mean_batch_objective": kwargs["training_loss"],
                                  "validation": self.validation}, allow_nan=False) + "\n")
        if self.progress:
            args = self.experiment_configuration.client.training_engine.arguments
            self.progress.epoch(epoch + 1, args["epochs"], self.training_examples, elapsed)

    def test_inference_start(self, current_time):
        if self.progress:
            self.progress.update(3, "held-out evaluation", "final epoch checkpoint selected by protocol")

    def test_inference_end(self, current_time, **kwargs):
        output = kwargs["output"]
        labels, logits, routes = output["labels"], output["logits"], output["routes"]
        predicted = logits.argmax(axis=1)
        self.test = class_metrics(labels, predicted)
        self.test["cross_entropy"] = mean_cross_entropy(labels, logits)
        self.test["route_counts"] = {
            "all": routing_counts(routes), "digit_4": routing_counts(routes[labels == 4]),
            "digit_9": routing_counts(routes[labels == 9]),
        }
        self.predictions = {"test_ids": np.arange(len(labels)), "labels": labels,
                            "logits": logits, "predicted": predicted, "expert": routes}

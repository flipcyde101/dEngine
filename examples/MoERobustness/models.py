"""Active-width matched classifiers for the fixed single-client pilot."""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from dengine.interfaces import ModuleBase
from dengine.models.decorators import register_model


class Prediction(NamedTuple):
    logits: Tensor
    expert: Tensor | None
    probabilities: Tensor | None


def residual_block():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64))


class DigitClassifier(ModuleBase):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(16 * 7 * 7, 64), nn.ReLU(),
        )
        # Create shared layers in the same order in both arms.
        self.head = nn.Linear(64, 10)

    def forward(self, images):
        return self.predict(images).logits


@register_model()
class DenseDigitClassifier(DigitClassifier):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.block = residual_block()

    def predict(self, images):
        hidden = self.features(images)
        return Prediction(self.head(hidden + self.block(hidden)), None, None)


@register_model()
class SparseDigitClassifier(DigitClassifier):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.experts = nn.ModuleList(residual_block() for _ in range(4))
        self.router = nn.Linear(64, len(self.experts))

    def predict(self, images):
        hidden = self.features(images)
        probabilities = self.router(hidden).softmax(dim=1)
        expert = probabilities.argmax(dim=1)
        update = torch.zeros_like(hidden)
        for slot, layer in enumerate(self.experts):
            selected = torch.nonzero(expert == slot, as_tuple=True)[0]
            if selected.numel():
                value = layer(hidden.index_select(0, selected))
                value = value * probabilities[selected, slot, None]
                update = update.index_copy(0, selected, value)
        return Prediction(self.head(hidden + update), expert, probabilities)


def balance_loss(prediction):
    if prediction.probabilities is None:
        return prediction.logits.new_zeros(())
    n_experts = prediction.probabilities.shape[1]
    assignments = torch.bincount(prediction.expert, minlength=n_experts)
    fractions = assignments / prediction.expert.numel()
    return n_experts * torch.dot(fractions, prediction.probabilities.mean(dim=0))

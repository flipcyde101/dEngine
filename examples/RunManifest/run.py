"""Run the manifest example with synthetic images and the dEngine CLI."""

import torch
from torch import nn

from dengine.bin.simulation import _cli
from dengine.dataset import SupervisedDataset
from dengine.dataset.decorators import register_dataset
from dengine.models.decorators import ModuleBase, register_model


@register_dataset("manifest_demo_images")
def demo_images(split: str, **kwargs) -> SupervisedDataset:
    labels = torch.arange(16) % 2
    pixels = torch.arange(64, dtype=torch.float32).reshape(16, 1, 2, 2)
    offset = 1 if split == "test" else 0
    return SupervisedDataset((pixels + offset).remainder(11) / 11, labels)


@register_model()
class ManifestDemoClassifier(ModuleBase):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))

    def forward(self, images):
        return self.layers(images)


if __name__ == "__main__":
    _cli()

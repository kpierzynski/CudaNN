import torch.nn as nn
import torch


class MLPModel(nn.Module):
    def __init__(self, in_features: int) -> None:
        super(MLPModel, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        data = data.view(data.shape[0], -1)  # flatten data

        return self.classifier(data)

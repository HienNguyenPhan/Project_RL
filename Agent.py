import torch.nn as nn

class Agent(nn.Module):
    def __init__(self, observation_shape, action_shape):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(observation_shape[-1], 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(16 * observation_shape[0] * observation_shape[1], 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, action_shape),
        )

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.permute(0, 3, 1, 2)  # Convert to NCHW
        x = self.cnn(x)
        x = self.flatten(x)
        return self.fc(x)
import torch
import torch.nn as nn
import torchvision

def get_resnet18_backbone(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

class AdvancedMLP(nn.Module):
    def __init__(self, input_dim=512, output_dim=4):
        super(AdvancedMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.network(x)

class TrajectoryGenerator(nn.Module):
    def __init__(self, input_dim=512, output_dim=50):
        super(TrajectoryGenerator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    def forward(self, x):
        return self.network(x)
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights

def get_resnet18(num_classes: int = 10, input_channels: int = 3):
    """
    ResNet-18 with adjustable input channels (e.g., 1 for MNIST).
    Uses the new weights=None argument instead of pretrained=False.
    """
    # Instantiate with no pre-trained weights
    model = models.resnet18(weights=None)

    # Adapt the first convolution for non-RGB inputs if needed
    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

    # Replace the final fully-connected layer for the desired number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

import torchvision.models as models
import torch.nn as nn

from utils import print_info


def build_model(pretrained=True, fine_tune=False, num_classes=10):
    if pretrained:
        print_info("Loading pre-trained weights")
    else:
        print_info("Not loading pre-trained weights")
    model = models.mobilenet_v3_small(pretrained=pretrained)

    if fine_tune:
        print_info("Fine-tuning all layers...")
        for param in model.parameters():
            param.requires_grad = True
    else:
        print_info("Freezing hidden layers...")
        for param in model.parameters():
            param.requires_grad = False

    model.classifier[3] = nn.Linear(in_features=1024, out_features=num_classes)
    return model

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models


def DeepLabv3(outputchannels=1):
    
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    model.classifier = DeepLabHead(2048, outputchannels)
    # Set the model in training mode
    model.train()
    #print(model)
    return model

DeepLabv3()
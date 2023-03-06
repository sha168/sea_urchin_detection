import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn

def create_model(num_classes, pretrained=None):
    # load Faster RCNN pre-trained model
    if pretrained == None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights='DEFAULT')
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if pretrained != None:
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))

    return model
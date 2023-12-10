import torch
import torch.nn as nn
from utils import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss() # for bbox
        self.bce = nn.BCEWithLogitsLoss() 
        self.cls_loss = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.const_cls = 1
        self.const_noobj = 10
        self.const_obj = 1
        self.const_bbox = 10

    def forward(self, predictions, target, anchors):
        obj = target[...,0] == 1
        noobj = target[...,0] == 0

        # No-obj loss
        no_obj_loss = self.bce((predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]))
        
        # Obj loss
        anchors = anchors.reshape(1,3,1,1,2)
        box_preds = torch.cat([self.sigmoid(predictions[...,1:3]), torch.exp(predictions[...,3:5]*anchors)], dim = -1)
        ious = intersection_over_union(box_preds[obj], target[...,1:5][obj]).detach()
        object_loss = self.bce((predictions[...,0:1][obj]), (ious*target[...,0:1][obj]))

        # Box Coord Loss
        predictions[...,1:3] = self.sigmoid(predictions[...,1:3])
        target[...,3:5] = torch.log(1e-8+target[...,3:5]/anchors)
        box_loss = self.mse(predictions[...,1:5][obj], target[...,1:5][obj])

        # Class Loss
        class_loss = self.bce((predictions[...,5:6][obj]),(target[...,5:6][obj]))

        total_loss = (self.const_bbox * box_loss)+(self.const_obj*object_loss)+(self.const_noobj*no_obj_loss)+(self.const_cls*class_loss)
        return total_loss 
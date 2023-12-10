import cv2
import numpy as np
import torch
import glob
from torch.utils.data import Dataset

def iou(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
        boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union

class BrainDataset(Dataset):
    def __init__(self, anchors, mode = "train", S=[13, 26, 52], image_size = 416, num_classes = 1, transform = None, root="/Users/sparrow/Desktop/yolo/"):
        self.iou_threshold = 0.5
        self.image_names = []
        self.label_names = []
        self.S = S
        self.num_classes = num_classes
        self.transform = transform
        self.anchors = torch.tensor(anchors[0]+anchors[1]+anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_anchors_per_scale = int(self.num_anchors/3)
        tmp_image_names = sorted(glob.glob(root+"dataset/images/"+mode+"/*"))
        tmp_labels_names = glob.glob(root+"dataset/labels/"+mode+"/*")
        for image_name in tmp_image_names:
            i_name = image_name.split("/")[-1].split(".")[0]
            check_label_file = root+"dataset/labels/"+mode+"/"+i_name+".txt"
            if check_label_file in tmp_labels_names:
                self.image_names.append(image_name)
                self.label_names.append(check_label_file)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image_path = self.image_names[index]
        label_path = self.label_names[index]
        image = cv2.imread(image_path)
        label_data = np.roll(np.loadtxt(label_path, delimiter = " ", ndmin=2), 4, axis=1).tolist() # X,Y,W,H,CLASS
        if self.transform:
            augmentations = self.transform(image=image, bboxes=label_data)
            image = augmentations["image"]
            bboxes = augmentations["bboxes"]
        
        targets = [torch.zeros((self.num_anchors//3, S, S, 6)) for S in self.S] 
        
        for box_elem in bboxes:
            iou_anchors = iou(torch.tensor(box_elem[2:4]), self.anchors) # send width and height to calc anchor's iou
            anchor_indices = iou_anchors.argsort(descending=True, dim = 0)
            x,y,w,h,class_label = box_elem
            has_anchor = [False]*3

            for anchor_idx in anchor_indices:
                branch_index = anchor_idx // self.num_anchors_per_scale # 0,1,2
                anchor_on_branch = anchor_idx % self.num_anchors_per_scale
                S = self.S[branch_index]
                j, i = int(S*y), int(S*x)
                anchor_taken = targets[branch_index][anchor_on_branch, j, i, 0]

                if not anchor_taken and not has_anchor[branch_index]:
                    targets[branch_index][anchor_on_branch, j, i, 0] = 1
                    x_cell, y_cell = S*x - i, S*y - j
                    w_cell, h_cell = S*w, S*h
                    box_coord = torch.tensor(
                        [x_cell, y_cell, w_cell, h_cell]
                    )
                    targets[branch_index][anchor_on_branch, j, i, 1:5] = box_coord
                    targets[branch_index][anchor_on_branch, j, i, 5] = int(class_label)
                    has_anchor[branch_index] = True
                elif not anchor_taken and iou_anchors[anchor_idx] > self.iou_threshold:
                    targets[branch_index][anchor_on_branch, j, i, 0] = -1
        return image, tuple(targets), image_path, label_path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.optim as optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import numpy as np
import cv2
import glob
import os
import argparse
from model import YOLOv3
from dataset import BrainDataset
from loss_fn import YoloLoss
from utils import non_max_suppression

parser = argparse.ArgumentParser(description = "pytorch model for Object Detection")
parser.add_argument('-lr', '--learning_rate', default = 4e-3)
parser.add_argument('-dim','--dim', default=416)
parser.add_argument('-ep', '--epoch', default = 50)
parser.add_argument('-rl', '--reload_epoch', default = 0)
parser.add_argument('-m', '--mode', required=True)
parser.add_argument('-root','--root', default="/Users/sparrow/Desktop/yolo/")
args = parser.parse_args()

LR = args.learning_rate
DIM = int(args.dim)
EPOCH = int(args.epoch)
MODE = args.mode
ROOT = args.root
RELOAD = int(args.reload_epoch)
yolo_v3_model = YOLOv3()
loss_fn =YoloLoss()

optimizer = optim.Adam(yolo_v3_model.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=3, gamma=0.05)

ANCHORS= [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
    ]
NMS_IOU_THRESH = 0.05
CONF_THRESHOLD = 0.40
S = [13, 26, 52]

scaled_anchors = torch.tensor(ANCHORS)*(torch.tensor([13, 26, 52]).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))


def cells_to_bboxes(predictions, anchors, S, is_preds=True):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
        .repeat(predictions.shape[0], 3, S, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()



def plot(image, bbox, color, type="mid"):
    try:
        image_shape = image.shape
        if color == "green":
            color = [0,255,0]
        elif color == "red":
            color = [0,0,255]
        elif color == "blue":
            color = [255,0,0]
        if type == "mid":
            cx,cy = int(bbox[2]*image_shape[1]), int(bbox[3]*image_shape[0])
            w,h = int(bbox[4]*image_shape[1]), int(bbox[5]*image_shape[0])
            x1,y1 = cx-int(w/2), cy-int(h/2)
            x2,y2 = cx+int(w/2), cy+int(h/2)
        else:
            x1,y1 = int(bbox[2]*image_shape[1]), int(bbox[3]*image_shape[0])
            x2,y2 = int(bbox[4]*image_shape[1]), int(bbox[4]*image_shape[0])
        start_point = (x1,y1)
        end_point = (x2,y2)
        image = cv2.rectangle(image, start_point, end_point, color, thickness=1)
    except:
        pass
    return image


def save_image(gt_image, pred_image, image_path):
    write_path = ROOT+"image_plots/"+image_path.split("/")[-1]
    gt_image = cv2.putText(gt_image, 'Ground Truth', (30,30), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,255,255), 2, cv2.LINE_AA)
    pred_image = cv2.putText(pred_image, 'Prediction', (30,30), cv2.FONT_HERSHEY_SIMPLEX ,  1, (255,255,255), 2, cv2.LINE_AA)
    image = cv2.hconcat([gt_image, pred_image])
    cv2.imwrite(write_path,image)


def train_yolo_v3():
    transform_train = A.Compose(
                [   A.Resize(height=DIM, width=DIM),
                    A.CLAHE(p=1.0),
                    A.HorizontalFlip(p=0.5),
                    A.ShiftScaleRotate(rotate_limit=20, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
                )
    clip_value = 3
    min_train_loss = 1000000
    epoch_tr_loss = []
    if RELOAD != 0:
        weight_path = glob.glob(ROOT+"weights/"+str(RELOAD)+".pth")
        yolo_v3_model.load_state_dict(torch.load(weight_path[0]))
    yolo_v3_model.train() # switch to train mode
    training_data = BrainDataset(ANCHORS, transform=transform_train, mode="train",root=ROOT)
    train_dataloader = DataLoader(training_data, batch_size=3, shuffle=True, pin_memory=True)
    if not os.path.exists(ROOT+"weights/"):
        os.system("mkdir "+ROOT+"weights/")
    for ep in range(RELOAD,EPOCH):
        train_losses = []
        with tqdm(train_dataloader, unit=" Train batch") as tepoch:
            tepoch.set_description(f"Train Epoch {ep+1}")
            for input_images, labels, image_paths, label_paths in tepoch:
                y0, y1, y2 = (
                                labels[0],
                                labels[1],
                                labels[2],
                            )
                with torch.cuda.amp.autocast():
                    out = yolo_v3_model(input_images)

                    loss = loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1]) + loss_fn(out[2], y2, scaled_anchors[2])
                    train_losses.append(loss.item())
                    
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(yolo_v3_model.parameters(), clip_value)
                optimizer.step()

        epoch_train_loss = np.mean(train_losses)
        epoch_tr_loss.append(epoch_train_loss)
        print(f'Epoch {ep+1}')
        print(f'train_loss : {epoch_train_loss}')
        if epoch_train_loss < min_train_loss:
            os.system("rm "+ROOT+"weights/*.pth")
            torch.save(yolo_v3_model.state_dict(), ROOT+"weights/"+str(ep+1)+".pth")
            min_train_loss = epoch_train_loss

def test_yolo_v3():
    weights_path = glob.glob(ROOT+"weights/*.pth")
    if len(weights_path) == 0:
        print("No weights file present in weights/")
    else:
        weights_path = weights_path[0] # takes the file to load
        print("Loading Weights: ",weights_path)
    transform_test = A.Compose(
                [   A.Resize(height=DIM, width=DIM),
                    A.CLAHE(p=1.0),
                    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255,),
                    ToTensorV2(),
                ], bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[]),
                )
    yolo_v3_model.load_state_dict(torch.load(weights_path))
    yolo_v3_model.eval()
    test_data = BrainDataset(ANCHORS , transform=transform_test, mode="test", root=ROOT)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True, pin_memory=True)
    test_losses = []
    # below code works only for batch size = 1 
    with tqdm(test_dataloader, unit=" Test batch") as tepoch:
        for input_images, labels, image_paths, label_paths in tepoch: 
            batch_size = input_images.shape[0]
            y0, y1, y2 = (
                labels[0],
                labels[1],
                labels[2],
            )
            with torch.no_grad():
                out = yolo_v3_model(input_images)
                loss = loss_fn(out[0], y0, scaled_anchors[0]) + loss_fn(out[1], y1, scaled_anchors[1]) + loss_fn(out[2], y2, scaled_anchors[2])
                test_losses.append(loss.item())
                bboxes = [[] for _ in range(batch_size)]
                for i in range(3):
                            S = out[i].shape[2]
                            anchor = torch.tensor([*ANCHORS[i]]) * S
                            boxes_scale_i = cells_to_bboxes(
                                out[i], anchor, S=S, is_preds=True
                            )
                            for idx, (box) in enumerate(boxes_scale_i):
                                bboxes[idx] += box



                # we just want one bbox for each label, not one for each scale
                # true_bboxes = cells_to_bboxes(labels[2], ANCHORS, S=S, is_preds=False)
                all_pred_boxes = [] # specified as [class_prediction, prob_score, cx, cy, w, h]
                # all_true_boxes = [] # specified as [class_prediction, prob_score, cx, cy, w, h]
                for idx in range(batch_size):
                    nms_boxes = non_max_suppression(
                        bboxes[idx],
                        iou_threshold=NMS_IOU_THRESH,
                        threshold=CONF_THRESHOLD,
                        box_format="midpoint",
                    )

                    for nms_box in nms_boxes:
                        all_pred_boxes.append(nms_box)

                    # for box in true_bboxes[idx]:
                    #     if box[1] > CONF_THRESHOLD:
                    #         all_true_boxes.append(box)

                for image_idx in range(len(image_paths)):
                    image_path = image_paths[image_idx]
                    label_path = label_paths[image_idx]
                    image = cv2.imread(image_path) # its a square dimensional image 256x256
                    image = cv2.resize(image, (DIM, DIM))
                    gt_image = image.copy()
                    pred_image = image
                    label_data = np.loadtxt(label_path, delimiter = " ", ndmin=2).tolist()
                    for iddx in range(len(label_data)):
                        label_data[iddx].insert(1,1)
                    # for gt in all_true_boxes:
                    #     gt_image = plot(gt_image, gt, "green")
                    for gt in label_data:
                        gt_image = plot(gt_image, gt, "green")                    
                    for pred in all_pred_boxes:
                        pred_image = plot(image, pred, "red")
                    save_image(gt_image, pred_image, image_path)



if MODE == "train":
    train_yolo_v3()
elif MODE == "test":
    test_yolo_v3()
else:
    print("Set --mode to train or test. Example: python3 train.py --mode train")
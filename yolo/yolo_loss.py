import torch
import torch.nn as nn


def _return_corner_coordinates(boxes, box_format):
    if box_format == "midpoint":
        x = boxes[..., 0:1]
        y = boxes[..., 1:2]
        boxes_width = boxes[..., 2:3]
        boxes_height = boxes[..., 3:4]
        box_x1 = x - boxes_width / 2
        box_y1 = y - boxes_height / 2
        box_x2 = x + boxes_width / 2
        box_y2 = y + boxes_height / 2
    elif box_format == "corners":
        box_x1 = boxes[..., 0:1]
        box_y1 = boxes[..., 1:2]
        box_x2 = boxes[..., 2:3]
        box_y2 = boxes[..., 3:4]
    return box_x1, box_y1, box_x2, box_y2


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    box1_x1, box1_y1, box1_x2, box1_y2 = _return_corner_coordinates(boxes_preds, box_format=box_format)
    box2_x1, box2_y1, box2_x2, box2_y2 = _return_corner_coordinates(boxes_labels, box_format=box_format)
    # Intersection rectangle corner coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    dx = x2 - x1
    dy = y2 - y1
    intersection_area = dx.clamp(0) * dy.clamp(0)  # in case they don't intersect
    box1_area = torch.abs((box1_x1 - box1_x2) * (box1_y1 - box1_y2))
    box2_area = torch.abs((box2_x1 - box2_x2) * (box2_y1 - box2_y2))
    union_area = box1_area + box2_area - intersection_area
    return intersection_area / union_area

def get_object_loss(target, bestbox, predictions, exists_box, s=-9):
    pred_box = bestbox * predictions[..., 25+s:26+s] + (1 - bestbox) * predictions[..., 20+s:21+s]
    mse = nn.MSELoss(reduction="sum")
    object_loss = mse(exists_box * pred_box, 
            exists_box * target[..., 20+s:21+s])
    return object_loss

def get_no_object_loss(exists_box, predictions, target, s=-9):
    mse = nn.MSELoss(reduction="sum")
    return (
        mse((1. - exists_box) * predictions[..., 20+s:21+s], (1. - exists_box) * target[..., 20+s:21+s]) +
        mse((1. - exists_box) * predictions[..., 25+s:26+s], (1. - exists_box) * target[..., 20+s:21+s])
    )

def get_class_loss(exists_box, predictions, target, s=-9):
    mse = nn.MSELoss(reduction="sum")
    return mse(exists_box * predictions[..., :20+s], exists_box * target[..., :20+s])



class YoloLoss(nn.Module):
    def __init__(self, grid_size=7, num_boxes=2, num_classes=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.lambda_coord = 5
        self.lambda_noobj = 0.5
        
    def forward(self, predictions, target):  # (N, S, S, C + B * 5)
        
        # Determine "responsible" bounding box predictor (based on the highest IoU)
        s = self.C - 20
        iou_b1 = intersection_over_union(predictions[..., 21+s:25+s], target[..., 21+s:25+s])
        iou_b2 = intersection_over_union(predictions[..., 26+s:30+s], target[..., 21+s:25+s])
        ious = torch.cat([iou_b1.unsqueeze(dim=0), iou_b2.unsqueeze(dim=0)], dim=0)
        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20+s].unsqueeze(3)  # I_obj (zero or one)
        
        # Box coordinates loss
        
        box_predictions = exists_box * (bestbox * predictions[..., 26+s:30+s] + (1 - bestbox) * predictions[..., 21+s:25+s])
        box_targets = exists_box * target[..., 21+s:25+s]
        # box_predictions = box_predictions1.clone()
        # box_targets = box_targets1.clone()
        
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        
        # (N, S, S, 4) -> (N*S*S, 4)
        box_loss = self.mse(box_predictions, box_targets)
        
        object_loss = get_object_loss(
            target=target,
            bestbox=bestbox,
            predictions=predictions,
            exists_box=exists_box,
            s=-9,
        )
        no_object_loss = get_no_object_loss(
            exists_box=exists_box,
            predictions=predictions,
            target=target,
            s=-9,
        )
        class_loss = get_class_loss(
            exists_box=exists_box,
            predictions=predictions,
            target=target,
            s=-9,
        )

        loss = (
            self.lambda_coord * box_loss +
            object_loss +
            self.lambda_noobj * no_object_loss + 
            class_loss
        )
        return loss

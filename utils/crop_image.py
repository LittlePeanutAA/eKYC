import numpy as np
import cv2
import torch


# Remove overlap boxes (NMS)
def non_max_suppression_fast(boxes, labels, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding box
        # and the smallest (x, y) coordinates for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs,
                         np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
                         )

    # return only the bounding boxes that were picked using the
    # integer data type

    final_labels = [labels[idx] for idx in pick]
    final_boxes = boxes[pick].astype("int")

    return final_boxes, final_labels


# Get the center of the box
def get_center_point(tensor):
    x_mid = (tensor[0] + tensor[2]) / 2  # (Xmin + Xmax)/2
    y_mid = (tensor[1] + tensor[3]) / 2  # (Ymin + Ymax)/2
    point = [x_mid, y_mid]
    return point


# Transform to perspective view
def perspective_transform(image, source_points):
    dest_points = np.float32([[0, 0], [500, 0], [500, 300], [0, 300]])
    M = cv2.getPerspectiveTransform(source_points, dest_points)
    dst = cv2.warpPerspective(image, M, (500, 300))

    return dst


# Crop image-part only including card
def CropImg(result, img):
    """
    CropImage(results, imgs):
    result = results.xyxy[i]
    result_pandas = results.pandas().xyxy[i]
    img = imgs[i]
    """
    # 1. Input overlap (Tensor 4*4, labels)
    # Tensor
    tensor = torch.tensor([])
    for i in range(len(result.xyxy)):
        tensor_i = result.xyxy[i][0:4].reshape(1, 4)
        tensor = torch.cat((tensor, tensor_i), 0)
    # Label
    label = result.cls

    # 2. Overlap
    final_boxes, final_labels = non_max_suppression_fast(tensor.numpy(), label.numpy(), 0.3)

    # 3. Tim midpoint cua cac boxes
    final_points = list(map(get_center_point, final_boxes))
    label_boxes = dict(zip(final_labels, final_points))

    # Define the tensor-to-string mappings
    # 4. Crop image
    print(label_boxes)

    if len(label_boxes) == 4:
        source_points = np.float32([label_boxes[2.0], label_boxes[3.0], label_boxes[1.0], label_boxes[0.0]])
        crop = perspective_transform(img, source_points)
        return crop
    else:
        return img

import cv2


def draw_bbox(image, tensors_box):
    for box_points in tensors_box:
        # Convert the tensor elements to integers
        point1 = (int(box_points[0]), int(box_points[1]))
        point2 = (int(box_points[2]), int(box_points[3]))

        # Draw the rectangle on the image
        cv2.rectangle(image, point1, point2, (0, 255, 0), 2)

    return image

#!/usr/bin/env python3
"""
Class Yolo that initiates, processes, filters boxes, and applies NMS.
"""
import numpy as np
import tensorflow as tf


class Yolo:
    """
    YOLOv3 model class for object detection.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo class instances.
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Processes raw Darknet model outputs into absolute boundary boxes.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            t_conf = output[..., 4:5]
            t_cls = output[..., 5:]

            s_xy = 1 / (1 + np.exp(-t_xy))
            box_conf = 1 / (1 + np.exp(-t_conf))
            box_cls_prob = 1 / (1 + np.exp(-t_cls))

            cx = np.tile(np.arange(grid_w), (grid_h, 1))
            cx = cx.reshape(grid_h, grid_w, 1, 1)

            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cy = cy.reshape(grid_h, grid_w, 1, 1)

            pw_ph = self.anchors[i]
            pw = pw_ph[:, 0].reshape(1, 1, num_anchors, 1)
            ph = pw_ph[:, 1].reshape(1, 1, num_anchors, 1)

            bx = (s_xy[..., 0:1] + cx) / grid_w
            by = (s_xy[..., 1:2] + cy) / grid_h
            bw = (pw * np.exp(t_wh[..., 0:1])) / input_w
            bh = (ph * np.exp(t_wh[..., 1:2])) / input_h

            x1 = bx - (bw / 2)
            y1 = by - (bh / 2)
            x2 = bx + (bw / 2)
            y2 = by + (bh / 2)

            img_h, img_w = image_size[0], image_size[1]
            processed_box = np.zeros(output[..., :4].shape)
            processed_box[..., 0] = x1[..., 0] * img_w
            processed_box[..., 1] = y1[..., 0] * img_h
            processed_box[..., 2] = x2[..., 0] * img_w
            processed_box[..., 3] = y2[..., 0] * img_h

            boxes.append(processed_box)
            box_confidences.append(box_conf)
            box_class_probs.append(box_cls_prob)

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Filters boundary boxes by box scores threshold.
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for i in range(len(boxes)):
            scores = box_confidences[i] * box_class_probs[i]
            classes = np.argmax(scores, axis=-1)
            scores_max = np.max(scores, axis=-1)

            filtering_mask = scores_max >= self.class_t

            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(classes[filtering_mask])
            box_scores.append(scores_max[filtering_mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Applies Non-max Suppression to the filtered bounding boxes.
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        # Find all unique class IDs that are present in our data
        unique_classes = np.unique(box_classes)

        # Process each class separately to satisfy NMS conditions per class
        for cls in unique_classes:
            # Mask to extract data only for the current class
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            # Sort indices of the current class by scores in descending order
            order = np.argsort(cls_scores)[::-1]

            # Keep checking boxes until none are left in the ordered list
            while order.size > 0:
                # Select index of the highest score box
                best_idx = order[0]
                box_predictions.append(cls_boxes[best_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best_idx])

                if order.size == 1:
                    break

                # Extract coordinates for the best box and the remaining boxes
                x1_best, y1_best, x2_best, y2_best = cls_boxes[best_idx]
                x1_rem = cls_boxes[order[1:], 0]
                y1_rem = cls_boxes[order[1:], 1]
                x2_rem = cls_boxes[order[1:], 2]
                y2_rem = cls_boxes[order[1:], 3]

                # Calculate coordinates of intersection rectangles
                inter_x1 = np.maximum(x1_best, x1_rem)
                inter_y1 = np.maximum(y1_best, y1_rem)
                inter_x2 = np.minimum(x2_best, x2_rem)
                inter_y2 = np.minimum(y2_best, y2_rem)

                # Compute area of intersection rectangles
                inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                # Compute area of both best box and remaining boxes
                best_area = (x2_best - x1_best) * (y2_best - y1_best)
                rem_area = (x2_rem - x1_rem) * (y2_rem - y1_rem)

                # Compute Intersection over Union (IoU)
                union_area = best_area + rem_area - inter_area
                iou = inter_area / union_area

                # Keep indices where IoU is less than the NMS threshold
                # This drops the boxes that overlap too much with the best box
                keep_indices = np.where(iou < self.nms_t)[0]

                # Update the order array
                # (offset by 1 because we compared order[1:])
                order = order[keep_indices + 1]

        # Convert lists of results back into standard numpy arrays
        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

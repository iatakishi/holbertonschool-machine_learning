#!/usr/bin/env python3
"""
Class Yolo that initiates and processes YOLOv3 model outputs and filters boxes.
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
            # Calculate total scores by
            # multiplying confidence and probabilities
            # shape: (grid_height, grid_width, anchor_boxes, classes)
            scores = box_confidences[i] * box_class_probs[i]

            # Find the index of the highest class probability for each box
            # shape: (grid_height, grid_width, anchor_boxes)
            classes = np.argmax(scores, axis=-1)

            # Extract the actual maximum score value for each box
            # shape: (grid_height, grid_width, anchor_boxes)
            scores_max = np.max(scores, axis=-1)

            # Create a boolean mask where
            # scores are greater than or equal to threshold
            filtering_mask = scores_max >= self.class_t

            # Filter and append data that satisfies the threshold condition
            filtered_boxes.append(boxes[i][filtering_mask])
            box_classes.append(classes[filtering_mask])
            box_scores.append(scores_max[filtering_mask])

        # Concatenate lists of arrays
        # into a single flat numpy array for each
        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

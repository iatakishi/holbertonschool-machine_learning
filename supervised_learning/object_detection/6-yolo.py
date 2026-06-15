#!/usr/bin/env python3
"""
Class Yolo that initiates, processes, filters, applies NMS,
loads, preprocesses images, and visualizes detected boxes.
"""
import glob
import os
import cv2
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

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            cls_mask = box_classes == cls
            cls_boxes = filtered_boxes[cls_mask]
            cls_scores = box_scores[cls_mask]

            order = np.argsort(cls_scores)[::-1]

            while order.size > 0:
                best_idx = order[0]
                box_predictions.append(cls_boxes[best_idx])
                predicted_box_classes.append(cls)
                predicted_box_scores.append(cls_scores[best_idx])

                if order.size == 1:
                    break

                x1_best, y1_best, x2_best, y2_best = cls_boxes[best_idx]
                x1_rem = cls_boxes[order[1:], 0]
                y1_rem = cls_boxes[order[1:], 1]
                x2_rem = cls_boxes[order[1:], 2]
                y2_rem = cls_boxes[order[1:], 3]

                inter_x1 = np.maximum(x1_best, x1_rem)
                inter_y1 = np.maximum(y1_best, y1_rem)
                inter_x2 = np.minimum(x2_best, x2_rem)
                inter_y2 = np.minimum(y2_best, y2_rem)

                inter_w = np.maximum(0.0, inter_x2 - inter_x1)
                inter_h = np.maximum(0.0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h

                best_area = (x2_best - x1_best) * (y2_best - y1_best)
                rem_area = (x2_rem - x1_rem) * (y2_rem - y1_rem)

                union_area = best_area + rem_area - inter_area
                iou = inter_area / union_area

                keep_indices = np.where(iou < self.nms_t)[0]
                order = order[keep_indices + 1]

        box_predictions = np.array(box_predictions)
        predicted_box_classes = np.array(predicted_box_classes)
        predicted_box_scores = np.array(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Loads all images from a specific folder path.
        """
        images = []
        if not folder_path.endswith('/'):
            folder_path += '/'
        image_paths = glob.glob(folder_path + '*')

        for path in image_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocesses a list of images for the YOLO model.
        """
        pimages = []
        image_shapes = []

        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for img in images:
            h, w, _ = img.shape
            image_shapes.append([h, w])

            resized_img = cv2.resize(img, (input_w, input_h),
                                     interpolation=cv2.INTER_CUBIC)

            rescaled_img = resized_img / 255.0
            pimages.append(rescaled_img)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
        """
        Displays the image with all standard boundary boxes drawn.
        """
        # Create a copy of the image to keep the original image pristine
        draw_img = image.copy()

        # Iterate over all detected boxes to draw rectangles and labels
        for i in range(len(boxes)):
            # Convert float coordinates to integers for OpenCV compatibility
            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])

            # Draw boundary box in blue (BGR: 255, 0, 0) with thickness 2
            cv2.rectangle(draw_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Retrieve text label and format score rounded to 2 decimals
            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]
            label_text = "{} {:.2f}".format(class_name, score)

            # Position label 5 pixels above the top-left corner
            label_pos = (x1, y1 - 5)

            # Draw text label in red (BGR: 0, 0, 255) using required options
            cv2.putText(draw_img, label_text, label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                        1, cv2.LINE_AA)

        # Display the image window using file_name as name
        cv2.imshow(file_name, draw_img)

        # Wait for key trigger event indefinitely
        key = cv2.waitKey(0)

        # Bitmask to catch key code natively (ord('s') matches ASCII value)
        if key & 0xFF == ord('s'):
            # Create detection directory path if it doesn't exist yet
            if not os.path.exists('detections'):
                os.makedirs('detections')

            # Save the annotated output frame inside detections folder
            cv2.imwrite(os.path.join('detections', file_name), draw_img)

        # Destroy window element to clean up memory correctly
        cv2.destroyAllWindows()

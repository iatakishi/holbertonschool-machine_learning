#!/usr/bin/env python3
"""
YOLOv3 alqoritmi ilə model çıxışlarını emal edən Yolo sinfi.
"""
import numpy as np
import tensorflow as tf


class Yolo:
    """
    YOLOv3 modelini başladan və çıxışlarını emal edən sinif.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo sinfinin konstruktoru.
        """
        self.model = tf.keras.models.load_model(model_path, compile=False)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def process_outputs(self, outputs, image_size):
        """
        Darknet modelindən gələn proqnozları emal edir.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Modelin giriş ölçülərini dinamik olaraq götürürük (adətən 416x416)
        input_w = self.model.input.shape[1]
        input_h = self.model.input.shape[2]

        for i, output in enumerate(outputs):
            grid_h, grid_w, num_anchors, _ = output.shape

            # Çıxış massivini hissələrə bölürük
            t_xy = output[..., :2]
            t_wh = output[..., 2:4]
            t_conf = output[..., 4:5]
            t_cls = output[..., 5:]

            # Aktivasiya funksiyalarını tətbiq edirik (Sigmoid)
            s_xy = 1 / (1 + np.exp(-t_xy))
            box_conf = 1 / (1 + np.exp(-t_conf))
            box_cls_prob = 1 / (1 + np.exp(-t_cls))

            # Tor xanalarının (grid) koordinat matrisini yaradırıq
            cx = np.tile(np.arange(grid_w), (grid_h, 1))
            cx = cx.reshape(grid_h, grid_w, 1, 1)

            cy = np.tile(np.arange(grid_h), (grid_w, 1)).T
            cy = cy.reshape(grid_h, grid_w, 1, 1)

            # Lövbər qutularının en və hündürlüklərini götürürük
            pw_ph = self.anchors[i]
            pw = pw_ph[:, 0].reshape(1, 1, num_anchors, 1)
            ph = pw_ph[:, 1].reshape(1, 1, num_anchors, 1)

            # Tor xanasına görə nisbi mərkəz koordinatları və ölçüləri
            bx = (s_xy[..., 0:1] + cx) / grid_w
            by = (s_xy[..., 1:2] + cy) / grid_h
            bw = (pw * np.exp(t_wh[..., 0:1])) / input_w
            bh = (ph * np.exp(t_wh[..., 1:2])) / input_h

            # Künc koordinatlarına keçid: (x1, y1, x2, y2)
            x1 = bx - (bw / 2)
            y1 = by - (bh / 2)
            x2 = bx + (bw / 2)
            y2 = by + (bh / 2)

            # Şəkil ölçülərinə uyğunlaşdırırıq (Scaling)
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

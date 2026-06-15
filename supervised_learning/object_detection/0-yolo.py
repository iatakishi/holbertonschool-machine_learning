#!/usr/bin/env python3
"""
YOLOv3 alqoritmi ilə obyektlərin tanınması üçün Yolo sinfi.
"""
import tensorflow as tf


class Yolo:
    """
    YOLOv3 modelini başladan və konfiqurasiya edən sinif.
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Yolo sinfinin konstruktoru.
        """
        # Darknet Keras modelini yükləyirik
        self.model = tf.keras.models.load_model(model_path, compile=False)

        # Sinif adlarını fayldan oxuyub list halına salırıq
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]

        # Digər ictimai (public) atributları təyin edirik
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

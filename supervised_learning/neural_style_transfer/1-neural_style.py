#!/usr/bin/env python3
"""Module that defines the NST class for Neural Style Transfer."""
import numpy as np
import tensorflow as tf


# Performs tasks for Neural Style Transfer
class NST:
    """Class that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                     'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor

        style_image - the image used as a style reference,
            stored as a numpy.ndarray
        content_image - the image used as a content reference,
            stored as a numpy.ndarray
        alpha - the weight for content cost
        beta - the weight for style cost
        """
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if (not isinstance(alpha, (int, float)) or alpha < 0):
            raise TypeError('alpha must be a non-negative number')
        if (not isinstance(beta, (int, float)) or beta < 0):
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels

        image - a numpy.ndarray of shape (h, w, 3) containing the
            image to be scaled

        Returns: the scaled image
        """
        if (not isinstance(image, np.ndarray) or
                image.ndim != 3 or image.shape[2] != 3):
            raise TypeError(
                'image must be a numpy.ndarray with shape (h, w, 3)')

        h, w, _ = image.shape
        if h > w:
            h_new = 512
            w_new = int(w * (512 / h))
        else:
            w_new = 512
            h_new = int(h * (512 / w))

        image = image[tf.newaxis, :]
        image = tf.image.resize(image, size=(h_new, w_new),
                                 method='bicubic')
        image = image / 255
        image = tf.clip_by_value(image, 0, 1)

        return image

    def load_model(self):
        """Creates the model used to calculate cost

        the model uses VGG19 Keras model as a base
        the model's input is the same as VGG19 input
        the model's output is a list containing the outputs of the
            VGG19 layers listed in style_layers followed by content_layer
        saves the model in the instance attribute model
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                           weights='imagenet')

        custom_objects = {'MaxPooling2D': tf.keras.layers.AveragePooling2D}
        vgg.save("vgg_base_model")
        vgg = tf.keras.models.load_model("vgg_base_model",
                                          custom_objects=custom_objects)

        style_outputs = [vgg.get_layer(name).output
                         for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)

        for layer in model.layers:
            layer.trainable = False

        self.model = model

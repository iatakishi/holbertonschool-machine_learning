#!/usr/bin/env python3
"""Module that defines the NST class for Neural Style Transfer."""
import numpy as np
import tensorflow as tf


class NST:
    """Class that performs tasks for neural style transfer"""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Class constructor"""
        if (not isinstance(style_image, np.ndarray) or
                style_image.ndim != 3 or style_image.shape[2] != 3):
            raise TypeError(
                'style_image must be a numpy.ndarray with shape (h, w, 3)')
        if (not isinstance(content_image, np.ndarray) or
                content_image.ndim != 3 or content_image.shape[2] != 3):
            raise TypeError(
                'content_image must be a numpy.ndarray with shape (h, w, 3)')
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError('alpha must be a non-negative number')
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError('beta must be a non-negative number')

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescales an image such that its pixels values are between
        0 and 1 and its largest side is 512 pixels
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
        Modifies VGG19 inplace to avoid disk saving permission errors.
        """
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet')

        # Replace MaxPooling2D layers with AveragePooling2D purely in memory
        x = vgg.input
        model_outputs = {}
        for layer in vgg.layers[1:]:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                x = tf.keras.layers.AveragePooling2D(
                    pool_size=layer.pool_size,
                    strides=layer.strides,
                    padding=layer.padding,
                    name=layer.name)(x)
            else:
                x = layer(x)
            model_outputs[layer.name] = x

        style_outputs = [model_outputs[name] for name in self.style_layers]
        content_output = model_outputs[self.content_layer]

        outputs = style_outputs + [content_output]

        model = tf.keras.models.Model(vgg.input, outputs)

        for layer in model.layers:
            layer.trainable = False

        self.model = model

    @staticmethod
    def gram_matrix(input_layer):
        """Calculates the gram matrix of a layer"""
        if (not isinstance(input_layer, (tf.Tensor, tf.Variable)) or
                len(input_layer.shape) != 4):
            raise TypeError('input_layer must be a tensor of rank 4')

        result = tf.linalg.einsum('bijc,bijd->bcd',
                                  input_layer, input_layer)
        input_shape = tf.shape(input_layer)
        num_locations = tf.cast(input_shape[1] * input_shape[2],
                                tf.float32)

        return result / num_locations

    def generate_features(self):
        """Extracts the features used to calculate neural style cost"""
        vgg19 = tf.keras.applications.vgg19

        preprocess_style = vgg19.preprocess_input(self.style_image * 255)
        preprocess_content = vgg19.preprocess_input(self.content_image * 255)

        style_features = self.model(preprocess_style)[:-1]
        content_feature = self.model(preprocess_content)[-1]

        self.gram_style_features = [
            self.gram_matrix(style_feature)
            for style_feature in style_features
        ]
        self.content_feature = content_feature

    def layer_style_cost(self, style_output, gram_target):
        """Calculates the style cost for a single layer"""
        if (not isinstance(style_output, (tf.Tensor, tf.Variable)) or
                len(style_output.shape) != 4):
            raise TypeError('style_output must be a tensor of rank 4')

        c = style_output.shape[-1]
        if (not isinstance(gram_target, (tf.Tensor, tf.Variable)) or
                gram_target.shape != (1, c, c)):
            raise TypeError(
                'gram_target must be a tensor of shape [1, {}, {}]'
                .format(c, c))

        gram_style = self.gram_matrix(style_output)

        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def style_cost(self, style_outputs):
        """Calculates the style cost for generated image"""
        length = len(self.style_layers)
        if (not isinstance(style_outputs, list) or
                len(style_outputs) != length):
            raise TypeError(
                'style_outputs must be a list with a length of {}'
                .format(length))

        weight = 1 / length
        style_cost = 0

        for style_output, gram_target in zip(style_outputs,
                                             self.gram_style_features):
            style_cost += weight * self.layer_style_cost(
                style_output, gram_target)

        return style_cost

    def content_cost(self, content_output):
        """Calculates the content cost for the generated image"""
        shape = self.content_feature.shape
        if (not isinstance(content_output, (tf.Tensor, tf.Variable)) or
                content_output.shape != shape):
            raise TypeError(
                'content_output must be a tensor of shape {}'
                .format(shape))

        return tf.reduce_mean(tf.square(content_output -
                                        self.content_feature))

    def total_cost(self, generated_image):
        """Calculates the total cost for the generated image"""
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'
                .format(shape))

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_style = self.style_cost(style_outputs)
        J_content = self.content_cost(content_output)
        J = self.alpha * J_content + self.beta * J_style

        return J, J_content, J_style

    def compute_grads(self, generated_image):
        """Calculates the gradients for the tf.Tensor generated image"""
        shape = self.content_image.shape
        if (not isinstance(generated_image, (tf.Tensor, tf.Variable)) or
                generated_image.shape != shape):
            raise TypeError(
                'generated_image must be a tensor of shape {}'
                .format(shape))

        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)

        gradients = tape.gradient(J_total, generated_image)

        return gradients, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01,
                       beta1=0.9, beta2=0.99):
        """Generates the neural style transfered image"""
        if not isinstance(iterations, int) or isinstance(iterations, bool):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be positive')

        if step is not None:
            if not isinstance(step, int) or isinstance(step, bool):
                raise TypeError('step must be an integer')
            if step <= 0 or step >= iterations:
                raise ValueError(
                    'step must be positive and less than iterations')

        if not isinstance(lr, (int, float)) or isinstance(lr, bool):
            raise TypeError('lr must be a number')
        if lr <= 0:
            raise ValueError('lr must be positive')

        if not isinstance(beta1, float):
            raise TypeError('beta1 must be a float')
        if beta1 < 0 or beta1 > 1:
            raise ValueError('beta1 must be in the range [0, 1]')

        if not isinstance(beta2, float):
            raise TypeError('beta2 must be a float')
        if beta2 < 0 or beta2 > 1:
            raise ValueError('beta2 must be in the range [0, 1]')

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             beta_1=beta1,
                                             beta_2=beta2)
        generated_image = tf.Variable(self.content_image)

        best_cost = float('inf')
        best_image = None

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(
                generated_image)

            if J_total < best_cost:
                best_cost = J_total.numpy()
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print("Cost at iteration {}: {}, content {}, style {}".format(
                    i, J_total.numpy(), J_content.numpy(), J_style.numpy()))

            if i < iterations:
                optimizer.apply_gradients([(grads, generated_image)])
                clipped_image = tf.clip_by_value(generated_image, 0.0, 1.0)
                generated_image.assign(clipped_image)

        # best_image[0] olaraq qaytarılır
        # çünki imshow() 4D array (1, h, w, 3) dəstəkləmir.
        return best_image[0], best_cost

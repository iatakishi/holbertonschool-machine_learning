def layer_style_cost(self, style_output, gram_target):
    """Calculates the style cost for a single layer

    style_output - tf.Tensor of shape (1, h, w, c) containing the
        layer style output of the generated image
    gram_target - tf.Tensor of shape (1, c, c) the gram matrix of
        the target style output for that layer

    Returns: the layer's style cost
    """
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

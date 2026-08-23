#!/usr/bin/env python3
"""Transformer model implementation for machine translation."""

import tensorflow as tf
import numpy as np


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-Head Attention layer."""

    def __init__(self, dm, h):
        """Initialize the layer."""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.dense = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (h, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        """Forward pass."""
        batch_size = tf.shape(q)[0]
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, _ = scaled_dot_product_attention(
            q, k, v, mask)
        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm))
        output = self.dense(concat_attention)
        return output


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights."""
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
    attention_weights = tf.nn.softmax(
        scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


class EncoderBlock(tf.keras.layers.Layer):
    """Encoder block with multi-head attention and FFN."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block."""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Forward pass."""
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Decoder block with masked MHA, MHA and FFN."""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Initialize the block."""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden, activation='relu'),
            tf.keras.layers.Dense(dm)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass."""
        attn1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.mha2(enc_output, enc_output, out1, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(out2 + ffn_output)
        return out3


class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding for the transformer."""

    def __init__(self, max_len, dm):
        """Initialize the layer."""
        super(PositionalEncoding, self).__init__()
        self.pe = np.zeros((1, max_len, dm))
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(
            np.arange(0, dm, 2) * -(np.log(10000.0) / dm))
        self.pe[0, :, 0::2] = np.sin(position * div_term)
        self.pe[0, :, 1::2] = np.cos(position * div_term)
        self.pe = tf.cast(self.pe, tf.float32)

    def call(self, x):
        """Forward pass."""
        return x + self.pe[:, :tf.shape(x)[1], :]


class Encoder(tf.keras.layers.Layer):
    """Encoder with N blocks."""

    def __init__(self, N, dm, h, hidden, max_len):
        """Initialize the encoder."""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(8194, dm)
        self.positional_encoding = PositionalEncoding(max_len, dm)
        self.blocks = [EncoderBlock(dm, h, hidden) for _ in range(N)]

    def call(self, x, training, mask):
        """Forward pass."""
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.positional_encoding(x)
        for i in range(self.N):
            x = self.blocks[i](x, training, mask)
        return x


class Decoder(tf.keras.layers.Layer):
    """Decoder with N blocks."""

    def __init__(self, N, dm, h, hidden, max_len):
        """Initialize the decoder."""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(8194, dm)
        self.positional_encoding = PositionalEncoding(max_len, dm)
        self.blocks = [DecoderBlock(dm, h, hidden) for _ in range(N)]

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        """Forward pass."""
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x = self.positional_encoding(x)
        for i in range(self.N):
            x = self.blocks[i](
                x, enc_output, training, look_ahead_mask, padding_mask)
        return x


class Transformer(tf.keras.Model):
    """Transformer model for machine translation."""

    def __init__(self, N, dm, h, hidden, max_len):
        """Initialize the transformer."""
        super(Transformer, self).__init__()
        self.encoder = Encoder(N, dm, h, hidden, max_len)
        self.decoder = Decoder(N, dm, h, hidden, max_len)
        self.linear = tf.keras.layers.Dense(8194)

    def call(self, inputs, target, training, enc_mask, dec_mask):
        """Forward pass."""
        enc_output = self.encoder(inputs, training, enc_mask)
        dec_output = self.decoder(
            target, enc_output, training, dec_mask, enc_mask)
        final_output = self.linear(dec_output)
        return final_output, None

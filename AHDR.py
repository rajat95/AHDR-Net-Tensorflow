#!/usr/bin/python
# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow.layers as layers


def conv2d(
    img,
    n_filters,
    rate=1,
    kernel_size=[3, 3],
    reuse=False,
    bias=True,
    activation='relu',
    ):
    return layers.conv2d(
        img,
        filters=n_filters,
        kernel_size=kernel_size,
        dilation_rate=rate,
        activation=activation,
        padding='same',
        reuse=reuse,
        use_bias=bias,
        kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
        )


def encoding(
    img,
    n_filters,
    rate=1,
    kernel_size=[3, 3],
    reuse=False,
    ):
    with tf.name_scope('shared_encoding'):
        net = conv2d(img, n_filters, rate, kernel_size, reuse=reuse)
        return net


def attn_module(
    z_i,
    z_r,
    n_filters,
    kernel_size,
    ):
    with tf.name_scope('attn_module'):
        z_s = tf.concat([z_i, z_r], axis=3)
        net = conv2d(z_s, n_filters=64, kernel_size=[3, 3])
        net = conv2d(net, n_filters=64, kernel_size=[3, 3],
                     activation=None)
        net = tf.sigmoid(net)
        return net


def attn_network(i_1, i_2, i_3):
    with tf.name_scope('attn_network'):
        z_1 = encoding(i_1, n_filters=64, kernel_size=[3, 3])
        z_r = encoding(i_2, n_filters=64, kernel_size=[3, 3],
                       reuse=True)
        z_3 = encoding(i_3, n_filters=64, kernel_size=[3, 3],
                       reuse=True)

        a_1 = attn_module(z_1, z_r, n_filters=64, kernel_size=[3, 3])
        a_3 = attn_module(z_3, z_r, n_filters=64, kernel_size=[3, 3])

        z_11 = tf.multiply(z_1, a_1)
        z_33 = tf.multiply(z_3, a_3)

        z_s = tf.concat([z_11, z_r, z_33], axis=3)
        return (z_s, z_r,a_1,a_3)


def dconv2d(x,n_filters=32,rate=2,kernel_size=[3, 3]):
    with tf.name_scope('dconv2d'):
        output = conv2d(x, n_filters=n_filters,
                        kernel_size=kernel_size, rate=rate)
        return output


def drdb(
    x,
    kernel_size=[3, 3],
    rate=2,
    growth_rate=32,
    ):
    with tf.name_scope('drdb'):
        x_1 = dconv2d(x, n_filters=growth_rate, rate=rate)
        x_2 = dconv2d(tf.concat([x, x_1], axis=3),
                      n_filters=growth_rate, rate=rate)
        x_3 = dconv2d(tf.concat([x, x_1, x_2], axis=3),
                      n_filters=growth_rate, rate=rate)
        x_4 = dconv2d(tf.concat([x, x_1, x_2, x_3], axis=3),
                       n_filters=growth_rate, rate=rate)
        x_5 = dconv2d(tf.concat([x, x_1, x_2, x_3, x_4], axis=3),
                       n_filters=growth_rate, rate=rate)

        output = tf.concat([
            x,
            x_1,
            x_2,
            x_3,
            x_4,
            x_5,
            ], axis=3)
        output = conv2d(output, n_filters=64, kernel_size=[1, 1])
        return x + output


def merging_network(z_s, z_r):
    with tf.name_scope('merging_network'):
        f_0 = conv2d(z_s, n_filters=64, kernel_size=[3, 3])
        f_1 = drdb(f_0, rate=2)
        f_2 = drdb(f_1, rate=2)
        f_3 = drdb(f_2, rate=2)
        f_4 = tf.concat([f_1, f_2, f_3], axis=3)
        f_5 = conv2d(f_4, n_filters=64, kernel_size=[3, 3])
        f_6 = conv2d(z_r + f_5, n_filters=64, kernel_size=[3, 3])
        f_7 = conv2d(f_6, n_filters=3, kernel_size=[3, 3])
        f_7 = tf.clip_by_value(f_7, 0.0, 1.0)
        return f_7


def ahdr_model(le, me, he):
    with tf.name_scope('ahdr_model'):
        (z_s, z_r,a_1,a_3) = attn_network(le, me, he)
        out = merging_network(z_s, z_r)
        return out


def build_ahdr(inputs):
    le = inputs[0]
    me = inputs[1]
    he = inputs[2]
    return ahdr_model(le, me, he)


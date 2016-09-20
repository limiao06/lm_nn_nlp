#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell


class TextRNN():
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, rnn_model, num_layers, l2_reg_lambda=0.0):
        if rnn_model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif rnn_model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif rnn_model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(rnn_model))

        cell = cell_fn(embedding_size)

        cell = rnn_cell.MultiRNNCell([cell] * num_layers)

        l2_loss = tf.constant(0.0)

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # embedding
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)

            # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
            inputs = tf.split(1, sequence_length, self.embedded_chars)
            self.embedded_chars_reshape = [tf.squeeze(input_, [1]) for input_ in inputs]

        # rnn
        outputs, last_state = rnn.rnn(cell, self.embedded_chars_reshape, dtype=tf.float32, scope='rnnLayer')

        # sentence_feature
        self.feature = outputs[-1]

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.feature, self.dropout_keep_prob)

        # softmax
        with tf.variable_scope('output'):
            W = tf.get_variable(
                "W",
                shape=[embedding_size, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            self.logits = tf.matmul(self.h_drop, W) + b
            self.probs = tf.nn.softmax(self.logits, name="probs")
            self.predictions = tf.argmax(self.logits, 1, name="predictions")

        for var in tf.trainable_variables():
            l2_loss += tf.nn.l2_loss(var)

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.logits, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

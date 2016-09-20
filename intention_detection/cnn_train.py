#! /usr/bin/env python

import sys
import copy
import json

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from text_cnn import TextCNN

# Parameters
# ==================================================

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "2,3,4,5", "Comma-separated filter sizes (default: '2,3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")
tf.flags.DEFINE_integer("vocab_min_freq", 5, "")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 512, "Batch Size (default: 512)")
tf.flags.DEFINE_integer("num_epochs", 6, "Number of training epochs (default: 6)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 1000, "Save model after this many steps (default: 1000)")
tf.flags.DEFINE_integer("max_dev_size", 50000, "Max number of dev set")
tf.flags.DEFINE_boolean("variables_average", False, "Allow device soft device placement")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

# Train Setting
tf.flags.DEFINE_string("train_data", "", "The training data")
tf.flags.DEFINE_string("model_dir", "", "The output model_dir")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


if not FLAGS.train_data or not FLAGS.model_dir:
	print("Error: train data or model dir not given!")
	exit(-1)

if not os.path.exists(FLAGS.model_dir):
    os.mkdir(FLAGS.model_dir)


gpu=0


# Data Preparatopn
# ==================================================

# Load data
def loadData(train_data):
    print("Loading data...")
    x_text, y, num = data_helpers.load_data_and_labels(train_data)
    print("Complete loading!")
    max_index, max_document_length = max([(i, len(xx.split(" "))) for i,xx in enumerate(x_text)], key=lambda x:x[1])
    print('Max document length: %d' %(max_document_length))
    print('Max length sent: %s' %(x_text[max_index]))
    x_text_unic = []
    for one in x_text:
        one_unic = unicode(one,'utf8',errors='ignore')
        x_text_unic.append(one_unic)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length, min_frequency=FLAGS.vocab_min_freq)
    x = np.array(list(vocab_processor.fit_transform(x_text_unic)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    sampleSize = len(y)
    devSize = sampleSize/10
    if devSize > FLAGS.max_dev_size:
        devSize = FLAGS.max_dev_size
    x_train, x_dev = x_shuffled[:-devSize], x_shuffled[-devSize:]
    y_train, y_dev = y_shuffled[:-devSize], y_shuffled[-devSize:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, x_dev, y_dev, vocab_processor, num

x_train, y_train, x_dev, y_dev, vocab_processor, num = loadData(FLAGS.train_data)

# Save configurations 
config_file = os.path.join(FLAGS.model_dir, 'config.json')
config_writer = file(config_file, 'w')
config_json = {}
for attr, value in sorted(FLAGS.__flags.items()):
    config_json[attr] = value
config_json['vocab_size'] = len(vocab_processor.vocabulary_)
json.dump(config_json, config_writer, indent=4)
config_writer.close()

log_file = os.path.join(FLAGS.model_dir, 'dev_performance.log')
logger = file(log_file, 'w')


# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        with tf.device('/gpu:%d' %(gpu)):
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=num,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)

            if FLAGS.variables_average:
                variable_averages = tf.train.ExponentialMovingAverage(0.9999, global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                opt_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
                train_op = tf.group(opt_op, variables_averages_op)
            else:
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.histogram_summary("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.scalar_summary("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.merge_summary(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = FLAGS.model_dir
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.scalar_summary("loss", cnn.loss)
            acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.merge_summary([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.merge_summary([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.train.SummaryWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.all_variables())

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.initialize_all_variables())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                logger.write("{}: step {}, loss {:g}, acc {:g}\n".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

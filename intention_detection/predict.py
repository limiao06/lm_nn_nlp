# coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import sys
import json
from tensorflow.contrib import learn
sys.path.append('/mnt/workspace/limiao/cnn/multi_gpu/')
import text_cnn

# only use for 2-class intention classification problem

# Parameters
# ==================================================

# Eval Parameters
tf.flags.DEFINE_string("model_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_string("model_point", "", "Checkpoint")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

grap = tf.Graph()
gpu = 0


class CNNPredict(object):
    def __init__(self, model_dir, model_point=None):
        vocab_path = os.path.join(model_dir, "vocab")
        if model_point:
            checkpoint_file = os.path.join(model_dir, 'checkpoints', 'model-'+model_point)
        else:
            checkpoint_file = tf.train.latest_checkpoint(model_dir+"/checkpoints")
        self.vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
        self.graph = tf.Graph()
        with self.graph.as_default():
            session_conf = tf.ConfigProto(
                    allow_soft_placement=FLAGS.allow_soft_placement,
                    log_device_placement=FLAGS.log_device_placement)
            self.sess = tf.Session(config=session_conf)
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(self.sess, checkpoint_file)
            self.input_x = self.graph.get_operation_by_name("input_x").outputs[0]
            self.dropout_keep_prob = self.graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            self.probs = self.graph.get_operation_by_name("output/probs").outputs[0]

    def predict_prob(self, sample):
        # Load data. Load your own data here
        # The sample should be unicode
        x_raw = []
        x_raw.append(sample)
        # Map data into vocabulary
        x_test = np.array(list(self.vocab_processor.transform(x_raw)))
        with self.graph.as_default(), tf.device('/gpu:%d' % gpu):
            probs = self.sess.run(probs, {self.input_x: x_test, self.dropout_keep_prob:1.0})
            return probs[0][1]


if __name__ == '__main__':
    cnn_predictor = CNNPredict(FLAGS.model_dir, FLAGS.model_point)
    print cnn_predictor.predict(u'刘德华')
    print cnn_predictor.predict(u'夜间 天气 怎么 样')
    print cnn_predictor.predict(u'真 的 没 心情 啊')

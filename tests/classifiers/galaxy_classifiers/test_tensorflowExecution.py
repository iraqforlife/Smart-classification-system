from unittest import TestCase

import tensorflow as tf
import numpy as np

class TestTensorFlow(TestCase):

    def setUp(self):
        pass

    def testTensorFlowCPU(self):
        # Creates a graph.
        with tf.device('/cpu:0'):
            self.a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
            self.b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
            c = tf.matmul(self.a, self.b)

            # Creates a session with log_device_placement set to True.
            sess = tf.Session()

            # Runs the op.
            print(sess.run(c))

            tf.assert_equal(c, tf.constant([22.0, 26.0, 49.0, 64.0], shape=[2, 2]))

    def testTensorFlowGPU(self):
        self.a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        self.b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(self.a, self.b)

        # Creates a session with log_device_placement set to True.
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

        # Runs the op.
        print(sess.run(c))

        tf.assert_equal(c, tf.constant([22.0, 26.0, 49.0, 64.0], shape=[2, 2]))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Course :
    GTI770 — Systèmes intelligents et apprentissage machine

Project :
    Lab # X - Lab's name

Students :
    Names — Permanent Code

Group :
    GTI770-H18-0X
"""

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import os


class MLPClassifierTensorBoard(object):
    def __init__(self, number_of_classes, batch_size, number_of_steps,learning_rate, image_size=424, number_of_channels=3,
                 dropout__keep_probability=0.5, train_path='/tmp'):

        """ Initialize the default parameters of a Multi-Layer Perceptron.

         Args:
            number_of_classes: The number of class the problem has.
            batch_size: The desired mini-batch size.
            image_size: The number of pixels in one dimension the image has (must be a square image).
            number_of_steps: The number of learning steps to run the training.
            learning_rate: The desired learning rate.
            train_path: The path in which the TensorBoard data will be saved.
        """

        self.number_of_classes = number_of_classes
        self.batch_size = batch_size
        self.image_size = image_size
        self.number_of_channels = number_of_channels
        self.number_of_steps = number_of_steps
        self.dropout_probability = dropout__keep_probability
        self.learning_rate = learning_rate
        self.train_path = train_path

    def train(self, dataset):
        with tf.Session(graph=tf.Graph()) as sess:

            with tf.name_scope("input"):
                X = tf.placeholder("float", [None, 74], name="X")
                y_ = tf.placeholder("float", [None, self.number_of_classes], name="y_ground_truth")
                keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")

            # Training loop
            for i in range(self.number_of_steps):
                    if i % 10 == 0:  # Record summaries and test-set accuracy
                        dict = feed_dict(False)

                        summary, train_accuracy, train_scores, train_prediction = sess.run(
                            [merged, accuracy, scores, predictions], feed_dict=dict)
                        test_writer.add_summary(summary, i)
                        print('Accuracy at step %s: %s' % (i, train_accuracy))

                    else:  # Record train set summaries, and train
                        if i % 100 == 99:  # Record execution stats
                            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                            run_metadata = tf.RunMetadata()
                            summary, _ = sess.run([merged, train_step],
                                                  feed_dict=feed_dict(True),
                                                  options=run_options,
                                                  run_metadata=run_metadata)
                            train_writer.add_run_metadata(run_metadata, '-step%04d' % i)
                            train_writer.add_summary(summary, i)
                            print('Adding run metadata for', i)

                        else:  # Record a summary
                            summary, _ = sess.run([merged, train_step], feed_dict=feed_dict(True))
                            train_writer.add_summary(summary, i)
            
            saver.save(sess, os.environ["VIRTUAL_ENV"] + "/data/models/exports/MLP/my_mlp/my_mlp_test")
            
            # Build the signature_def_map.
            classification_inputs = tf.saved_model.utils.build_tensor_info(X)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(scores)
            classification_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                            classification_inputs
                    },
                    outputs={
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                            classification_outputs_classes,
                        tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                            classification_outputs_scores
                    },
                    method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
            tensor_info_x = tf.saved_model.utils.build_tensor_info(X)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

            prediction_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={'images': tensor_info_x},
                    outputs={'classes': tensor_info_y},
                    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.TRAINING],
                signature_def_map={
                    'predict_images':
                        prediction_signature,
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        classification_signature,
                },
                legacy_init_op=legacy_init_op)

            # Export model.
            builder.save()
            print("Model saved and exported.")

            train_writer.close()
            test_writer.close()

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import pickle
import time

import cv2
import numpy as np
import tensorflow as tf

from auxiliary.settings import initialize_dataset_config, FCN_INPUT_SIZE, PATH_TO_MODEL
from auxiliary.utils import load_data, angular_error, print_angular_errors, get_visualization, LowestTrigger
from classes.data.DataProvider import DataProvider

GLOBAL_WEIGHT_DECAY = 5.7e-5
DROPOUT = 0.5
TRAINING_BATCH_SIZE = 16
FC1_SIZE = 64
FC1_KERNEL_SIZE = 6

TEST_PERIOD = 20
VISUALIZATION_SIZE = 512
WRITE_SUMMARY = True
IMAGE_SUMMARY_INT = 10

FINE_TUNE_LR_RATIO = 1e-1
LEARNING_RATE = 3e-4
LR_DECAY = 1
LR_DECAY_INTERVAL = 100

CKPTS_TO_KEEP = 0
CKPT_PERIOD = 0

TRAINING_VISUALIZATION = 200

# Visualization
MERGED_IMAGE_SIZE = 400

slim = tf.contrib.slim


# noinspection PyUnresolvedReferences
class FullyConvNet:

    def __init__(self, sess=None, name=None, kwargs=None):
        if kwargs is None:
            kwargs = {}
        global TRAINING_FOLDS, TEST_FOLDS
        self.name = name
        self.wd = GLOBAL_WEIGHT_DECAY
        TRAINING_FOLDS, TEST_FOLDS = initialize_dataset_config(**kwargs)
        self.training_data_provider = None
        self.sess = sess

        self.dropout, self.learning_rate, self.illuminants, self.images = None, None, None, None

        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_regularizer=slim.l2_regularizer(GLOBAL_WEIGHT_DECAY)):
            self.build()

        tf.global_variables_initializer()

        self.test_nets = {}
        self.saver = tf.train.Saver(max_to_keep=CKPTS_TO_KEEP)

    @staticmethod
    def build_branches(images, dropout):
        images = tf.clip_by_value(images, 0.0, 65535.0) * (1.0 / 65535)

        feed_to_fc = []

        with tf.variable_scope('AlexNet'):
            alex_images = (tf.pow(images, 1.0 / INPUT_GAMMA) * 255.0)[:, :, :, ::-1]
            alex_outputs = create_convnet(alex_images)

        feed_to_fc.append(alex_outputs['features_out'])
        feed_to_fc = tf.concat(axis=3, values=feed_to_fc)
        print("Feed to FC shape", feed_to_fc.get_shape())

        fc1 = slim.conv2d(feed_to_fc, FC1_SIZE, [FC1_KERNEL_SIZE, FC1_KERNEL_SIZE], scope='fc1')
        fc1 = slim.dropout(fc1, dropout)
        print("FC1 shape", fc1.get_shape())

        fc2 = slim.conv2d(fc1, 3, [1, 1], scope='fc2', activation_fn=None)
        print("FC2 shape", fc2.get_shape())

        return fc2

    def build(self):
        self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
        self.illuminants = tf.placeholder(tf.float32, shape=(None, 3), name='illuminants')
        self.images = tf.placeholder(tf.float32, shape=(None, FCN_INPUT_SIZE, FCN_INPUT_SIZE, 3), name='images')

        with tf.variable_scope('FullyConvNet'):
            fc2 = self.build_branches(self.images, self.dropout)

        self.per_patch_estimate = fc2
        self.illum_normalized = tf.nn.l2_normalize(tf.reduce_sum(fc2, axis=(1, 2)), 1)
        self.train_visualization = get_visualization(self.images,
                                                     self.per_patch_estimate,
                                                     self.illum_normalized,
                                                     self.illuminants,
                                                     (VISUALIZATION_SIZE, VISUALIZATION_SIZE))

        self.loss = self.get_angular_loss(tf.reduce_sum(fc2, axis=(1, 2)), self.illuminants, LENGTH_REGULARIZATION)
        self.scalar_summaries = tf.summary.merge([tf.summary.scalar('loss', self.loss)])

        reg_losses = tf.add_n(slim.losses.get_regularization_losses())
        self.total_loss = self.loss + reg_losses
        self.train_step_adam = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss)

        var_list1 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FullyConvNet/AlexNet')
        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FullyConvNet/fc1') + \
                    tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='FullyConvNet/fc2')

        for v in var_list1:
            print('list1', v.name)
        for v in var_list2:
            print('list2', v.name)

        opt1 = tf.train.AdamOptimizer(self.learning_rate * FINE_TUNE_LR_RATIO)
        opt2 = tf.train.AdamOptimizer(self.learning_rate)
        grads = tf.gradients(self.total_loss, var_list1 + var_list2)
        grads1, grads2 = grads[:len(var_list1)], grads[len(var_list1):]
        train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
        train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
        self.train_step_sgd = tf.group(train_op1, train_op2)

    def get_summary_variables(self, i, j):
        summary_variables = []
        if i >= 2 and WRITE_SUMMARY and j == 0:
            summary_variables.append(self.scalar_summaries)
            if i % IMAGE_SUMMARY_INT == 0:
                pass
        return summary_variables

    def train(self, epochs):
        trigger = LowestTrigger()
        os.mkdir(self.get_ckpt_folder())

        path_to_train = os.path.join(self.get_ckpt_folder(), "training")
        train_writer = tf.summary.FileWriter(logdir=path_to_train, graph=self.sess.graph)

        path_to_val = os.path.join(self.get_ckpt_folder(), "val")
        val_writer = tf.summary.FileWriter(logdir=path_to_val)

        print("TF", TRAINING_FOLDS)
        training_batch_size = TRAINING_BATCH_SIZE
        self.training_data_provider = DataProvider(True, TRAINING_FOLDS)
        self.training_data_provider.set_batch_size(training_batch_size)
        test_summary_input = tf.placeholder(tf.float32, shape=())
        test_summary = tf.summary.scalar('loss', test_summary_input)
        batches_per_training_epoch = self.training_data_provider.data_count // training_batch_size

        for i in range(1, epochs + 1):
            learning_rate = LEARNING_RATE * pow(LR_DECAY, 1.0 * i / LR_DECAY_INTERVAL)
            epoch_starting_time = time.time()
            training_losses = []
            for j in range(batches_per_training_epoch):
                batch = self.training_data_provider.get_batch()
                summary_variables = self.get_summary_variables(i, j)

                # Visualize some training images for monitoring the process
                visualization = []
                if TRAINING_VISUALIZATION and i % TRAINING_VISUALIZATION == 0:
                    visualization.append(self.train_visualization)

                loss, _, global_loss, summary, ppest, vis = self.sess.run([self.loss, self.train_step_adam(i),
                                                                           self.loss, summary_variables,
                                                                           self.per_patch_estimate, visualization],
                                                                          feed_dict={
                                                                              self.images: batch[0],
                                                                              self.illuminants: batch[2],
                                                                              self.dropout: DROPOUT,
                                                                              self.learning_rate: learning_rate
                                                                          })
                for s in summary:
                    train_writer.add_summary(s, i)

                if vis:
                    folder = os.path.join(self.get_ckpt_folder(), "training_visualization")
                    os.mkdir(folder)
                    for k, merged in enumerate(vis[0]):
                        summary_fn = '{}/{:04d}-{:03d}.jpg'.format(folder, i, j * len(vis) + k)
                        cv2.imwrite(summary_fn, merged[:, :, ::-1] * 255)

                training_losses.append(loss)

            training_loss = sum(training_losses) / len(training_losses)
            print("*{}* E {:4d}, TL {:.3f}, VL {:.3f}, D {:.3f}, t {:4.1f}".format(self.name, i, training_loss, 10,
                                                                                   10 - training_loss,
                                                                                   time.time() - epoch_starting_time))
            saved = False
            if CKPT_PERIOD and i % CKPT_PERIOD == 0:
                self.save(i)
                saved = True

            if TEST_PERIOD and i % TEST_PERIOD == 0:
                summary = i // TEST_PERIOD % 5 == 0
                errors = self.test(summary=summary, scales=[0.5], summary_key=i)[0]
                val_writer.add_summary(test_summary.eval(feed_dict={test_summary_input: np.mean(errors)}), i)

                if trigger.push(np.mean(errors)):
                    error_fn = self.get_ckpt_folder() + 'error{:04d}.pkl'.format(i)
                    pickle.dump(errors, open(error_fn, 'wb'), protocol=-1)
                    if not saved:
                        self.save(i)
                        saved = True
                print('mean(errors) from fcn.py line: 330', np.mean(errors))

        self.training_data_provider.stop()

    def test_network(self):
        records = load_data(['m'])

        for r in records:
            scale = 1
            img = np.clip((r.img / r.img.max()), 0, 1)
            img = np.power(img, 2.2)
            img[:, :img.shape[1] // 2:, 0] *= 3
            if scale != 1.0:
                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
            shape = img.shape[:2]
            if shape not in self.test_nets:
                test_net = {
                    "images": tf.placeholder(tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
                }
                with tf.variable_scope("FullyConvNet", reuse=True):
                    test_net['pixels'] = FullyConvNet.build_branches(test_net['images'], 1.0)
                self.test_nets[shape] = test_net
            test_net = self.test_nets[shape]

            pixels = self.sess.run(test_net['pixels'], feed_dict={test_net['images']: img[None, :, :, :]})
            pixels = pixels[0].astype(np.float32)
            pixels /= np.linalg.norm(pixels, axis=2)[:, :, None]

            cv2.imshow('pixels', cv2.resize(pixels, (0, 0), fx=10, fy=10))
            cv2.imshow('image', cv2.resize(img, (0, 0), fx=0.5, fy=0.5))
            cv2.waitKey(0)

    def test(self, summary=False, scales=None, summary_key=0, data=None):
        if scales is None:
            scales = [1.0]
        if not TEST_FOLDS:
            return [0]
        records = load_data(TEST_FOLDS) if data is None else data

        t = time.time()

        weights = [1.0] * len(scales)
        summaries, outputs, ground_truth, avg_confidence, errors = [], [], [], [], []
        for r in records:
            all_pixels = []
            for scale, weight in zip(scales, weights):
                img = r.img
                if scale != 1.0:
                    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)
                shape = img.shape[:2]
                if shape not in self.test_nets:
                    aspect_ratio = 1.0 * shape[1] / shape[0]
                    if aspect_ratio < 1:
                        target_shape = (400, 400 * aspect_ratio)
                    else:
                        target_shape = (400 / aspect_ratio, 400)
                    target_shape = tuple(map(int, target_shape))

                    test_net = {
                        "illuminants": tf.placeholder(tf.float32, shape=(None, 3), name='test_illums'),
                        "images": tf.placeholder(tf.float32, shape=(None, shape[0], shape[1], 3), name='test_images')
                    }

                    with tf.variable_scope("FullyConvNet", reuse=True):
                        test_net['pixels'] = FullyConvNet.build_branches(test_net['images'], 1.0)
                        test_net['est'] = tf.reduce_sum(test_net['pixels'], axis=(1, 2))
                    test_net['merged'] = get_visualization(test_net['images'], test_net['pixels'], test_net['est'],
                                                           test_net['illuminants'], target_shape)
                    self.test_nets[shape] = test_net
                test_net = self.test_nets[shape]

                pixels, est, merged = self.sess.run([test_net['pixels'], test_net['est'], test_net['merged']],
                                                    feed_dict={test_net['images']: img[None, :, :, :],
                                                               test_net['illuminants']: r.illuminant[None, :]})

                pixels, merged = pixels[0], merged[0]
                all_pixels.append(weight * pixels.reshape(-1, 3))

            all_pixels = np.sum(np.concatenate(all_pixels, axis=0), axis=0)
            est = all_pixels / (np.linalg.norm(all_pixels) + 1e-7)
            outputs.append(est)
            ground_truth.append(r.illuminant)
            error = math.degrees(angular_error(est, r.illuminant))
            errors.append(error)
            avg_confidence.append(np.mean(np.linalg.norm(all_pixels)))
            summaries.append((r.file_name, error, merged))

        print("Full Image:")
        ret = print_angular_errors(errors)
        ppt = (time.time() - t) / len(records)
        print('Test time:', time.time() - t, 'per image:', ppt)

        if summary:
            for file_name, error, merged in summaries:
                folder = os.path.join(self.get_ckpt_folder(), "test{:04d}summaries_{:4f}".format(summary_key, scale))
                os.mkdir(folder)
                summary_fn = '{}/{:5.3f}-{}.png'.format(folder, error, file_name)
                cv2.imwrite(summary_fn, merged[:, :, ::-1] * 255)

        return errors, ppt, outputs, ground_truth, ret, avg_confidence

    def get_ckpt_folder(self, name: str = None) -> str:
        return os.path.join(PATH_TO_MODEL, self.name if name is None else name)

    def get_filename(self, key, name=None) -> str:
        return os.path.join(self.get_ckpt_folder(self.name if name is None else name), str(key) + '.ckpt')

    def save(self, key):
        print("Model saved in file: {}".format(self.saver.save(self.sess, self.get_filename(key))))

    def load(self, key, name=None):
        self.load_absolute(self.get_filename(key, name=name))

    def load_absolute(self, file_name):
        self.saver.restore(self.sess, file_name)
        print("Model {} restored.".format(file_name))

    @staticmethod
    def get_angular_loss(vec1, vec2):
        with tf.name_scope('angular_error'):
            safe_v = 0.999999
            if len(vec1.get_shape()) == 2:
                illum_normalized = tf.nn.l2_normalize(vec1, 1)
                _illum_normalized = tf.nn.l2_normalize(vec2, 1)
                dot = tf.reduce_sum(illum_normalized * _illum_normalized, 1)
                dot = tf.clip_by_value(dot, -safe_v, safe_v)
                length_loss = tf.reduce_mean(tf.maximum(tf.log(tf.reduce_sum(vec1 ** 2, axis=1) + 1e-7), 0))
            else:
                assert len(vec1.get_shape()) == 4
                illum_normalized = tf.nn.l2_normalize(vec1, 3)
                _illum_normalized = tf.nn.l2_normalize(vec2, 3)
                dot = tf.reduce_sum(illum_normalized * _illum_normalized, 3)
                dot = tf.clip_by_value(dot, -safe_v, safe_v)
                length_loss = tf.reduce_mean(tf.maximum(tf.log(tf.reduce_sum(vec1 ** 2, axis=3) + 1e-7), 0))
            angle = tf.acos(dot) * (180 / math.pi)
            return tf.reduce_mean(angle) + length_loss

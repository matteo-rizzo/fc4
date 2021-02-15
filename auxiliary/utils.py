import math
import os
import pickle
import sys
from math import *

import cv2
import numpy as np
import tensorflow as tf

from auxiliary.settings import VIS_GAMMA
from classes.data.datasets.ChengDataset import ChengDataset
from classes.data.datasets.GehlerDataset import GehlerDataset

UV_SCALE = 0.75


def get_image_pack_fn(key):
    ds = key[0]
    if ds == 'g':
        fold = int(key[1])
        return GehlerDataset().get_image_pack_fn(fold)
    elif ds == 'c':
        camera = int(key[1])
        fold = int(key[2])
        return ChengDataset(camera).get_image_pack_fn(fold)
    elif ds == 'm':
        assert False


def load_data(folds):
    records = []
    for fold in folds:
        file_name = get_image_pack_fn(fold)
        print("Loading image pack: {}".format(file_name))

        # Cached
        if file_name not in load_data.data:
            load_data.data[file_name] = pickle.load(open(file_name))
        records += load_data.data[file_name]
    return records


load_data.data = {}


def set_target_gpu(gpus):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpus))


def angular_error(estimation, ground_truth):
    return acos(
        np.clip(np.dot(estimation, ground_truth) / np.linalg.norm(estimation) / np.linalg.norm(ground_truth), -1, 1))


def summary_angular_errors(errors):
    errors = sorted(errors)

    def g(f):
        return np.percentile(errors, f * 100)

    median = g(0.5)
    mean = np.mean(errors)
    trimean = 0.25 * (g(0.25) + 2 * g(0.5) + g(0.75))
    results = {
        '25': np.mean(errors[:int(0.25 * len(errors))]),
        '75': np.mean(errors[int(0.75 * len(errors)):]),
        '95': g(0.95),
        'tri': trimean,
        'med': median,
        'mean': mean
    }
    return results


def just_print_angular_errors(results):
    print("25: %5.3f," % results['25'], end=' ')
    print("med: %5.3f" % results['med'], end=' ')
    print("tri: %5.3f" % results['tri'], end=' ')
    print("avg: %5.3f" % results['mean'], end=' ')
    print("75: %5.3f" % results['75'], end=' ')
    print("95: %5.3f" % results['95'])


def print_angular_errors(errors):
    print("%d images tested. Results:" % len(errors))
    results = summary_angular_errors(errors)
    just_print_angular_errors(results)
    return results


class LowestTrigger:

    def __init__(self):
        self.minimum = None

    def push(self, value):
        if self.minimum is None or value < self.minimum:
            self.minimum = value
            return True
        return False


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle (in degrees).
    The returned image will be large enough to hold the entire new image, with a black background
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]])

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([[1, 0, int(new_w * 0.5 - image_w2)], [0, 1, int(new_h * 0.5 - image_h2)], [0, 0, 1]])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    return cv2.warpAffine(image, affine_mat, (new_w, new_h), flags=cv2.INTER_LINEAR)


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (bb_w - 2 * x, bb_h - 2 * y)


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if width > image_size[0]:
        width = image_size[0]

    if height > image_size[1]:
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]


def rotate_and_crop(image, angle):
    image_width, image_height = image.shape[:2]
    image_rotated = rotate_image(image, angle)
    image_rotated_cropped = crop_around_center(image_rotated,
                                               *largest_rotated_rect(image_width, image_height, math.radians(angle)))
    return image_rotated_cropped


class Tee(object):

    def __init__(self, name):
        self.file = open(name, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.file.flush()
        self.stdout.flush()

    def write_to_file(self, data):
        self.file.write(data)


def hdr2ldr(raw):
    return (np.clip(np.power(raw / (raw.max() * 0.5), 1 / 2.2), 0, 1) * 255).astype(np.uint8)


def bgr2uvl(raw):
    u = np.log(raw[:, :, 2] / raw[:, :, 1])
    v = np.log(raw[:, :, 0] / raw[:, :, 1])
    l = np.log(0.2126 * raw[:, :, 2] + 0.7152 * raw[:, :, 1] + 0.0722 * raw[:, :, 0])
    l = (l - l.mean()) * 0.3 + 0.5
    u = u * UV_SCALE + 0.5
    v = v * UV_SCALE + 0.5
    uvl = np.stack([u, v, l], axis=2)
    uvl = (np.clip(uvl, 0, 1) * 255).astype(np.uint8)
    return uvl


def bgr2nrgb(raw):
    rgb = raw / np.maximum(1e-4, np.linalg.norm(raw, axis=2, keepdims=True))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def get_WB_image(img, illuminant):
    return img / illuminant[::-1]


def slice_list(l, fractions):
    sliced = []
    for i in range(len(fractions)):
        total_fraction = sum(fractions)
        start = int(round(1.0 * len(l) * sum(fractions[:i]) / total_fraction))
        end = int(round(1.0 * len(l) * sum(fractions[:i + 1]) / total_fraction))
        sliced.append(l[start:end])
    return sliced


def get_session():
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def _activation_summary(x):
    """Helper to create summaries for activations.

      Creates a summary that provides a histogram of activations.
      Creates a summary that measure the sparsity of activations.

      Args:
        x: Tensor
      Returns:
        nothing
      """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    if tensor_name in _activation_summary.summarized:
        return
    _activation_summary.summarized.append(tensor_name)
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


_activation_summary.summarized = []


# Output: RGB images
def get_visualization(images, illums_est, illums_pooled, illums_ground,
                      target_shape):
    confidence = tf.sqrt(tf.reduce_sum(illums_est ** 2, axis=3))

    vis_confidence = confidence[:, :, :,
                     None]  # / tf.reduce_max(confidence, axis=(1, 2), keep_dims=True)[:,:,:,None]

    color_thres = [tf.constant(250.0 * i) for i in range(1, 5)]
    mean_confidence_value = tf.reduce_mean(confidence, axis=(0, 1, 2))
    vis_confidence_colored = tf.cond(mean_confidence_value < color_thres[0],
                                     lambda: vis_confidence * np.array((0, 0, 1)).reshape(1, 1, 1, 3) / 500.0,
                                     lambda: tf.cond(mean_confidence_value < color_thres[1],
                                                     lambda: vis_confidence * np.array((0, 1, 1)).reshape(1, 1, 1,
                                                                                                          3) / 1000.0,
                                                     lambda: tf.cond(mean_confidence_value < color_thres[2],
                                                                     lambda: vis_confidence * np.array(
                                                                         (0, 1, 0)).reshape(1, 1, 1, 3) / 2000.0,
                                                                     lambda: tf.cond(
                                                                         mean_confidence_value < color_thres[3],
                                                                         lambda: vis_confidence * np.array(
                                                                             (1, 1, 0)).reshape(1, 1, 1, 3) / 3000.0,
                                                                         lambda: vis_confidence * np.array(
                                                                             (1, 0, 0)).reshape(1, 1, 1, 3) / 4000.0
                                                                     )
                                                                     )
                                                     )
                                     )

    vis_est = tf.nn.l2_normalize(illums_est, 3)

    exposure_boost = 5

    img = tf.pow(images[:, :, :, ::-1] / 65535 * exposure_boost, 1 / VIS_GAMMA)
    img_corrected = tf.pow(
        images[:, :, :, ::-1] / 65535 / illums_pooled[:, None, None, :] * exposure_boost *
        tf.reduce_mean(illums_pooled, axis=(1), keep_dims=True)[:, None, None, :],
        1 / VIS_GAMMA)

    visualization = [
        img,
        img_corrected,
        vis_confidence_colored,
        vis_confidence * vis_est,
        vis_est,
        # tf.nn.l2_normalize(illums_ground, 1)[:, None, None, :],
        tf.nn.l2_normalize(illums_pooled, 1)[:, None, None, :]
    ]

    fcn_padding = 0  # = int(224.0 / int(images.get_shape()[1])  * target_shape[0]) // 2 # For receptive field offsets

    ##################
    confidence_dist = confidence[:, :, :, None] / tf.reduce_sum(
        confidence, axis=(1, 2), keep_dims=True)[:, :, :, None]
    mean_est = tf.reduce_mean(vis_est, axis=(1, 2), keep_dims=True)
    sq_deviation = tf.pow(vis_est - mean_est, 2)
    weighted_sq_dev = confidence_dist * sq_deviation
    variance = tf.reduce_sum(weighted_sq_dev, axis=(1, 2))
    ##################

    for i in range(len(visualization)):
        vis = visualization[i]
        if i == 0:
            padding = 0
        else:
            padding = fcn_padding
        if int(vis.get_shape()[3]) == 1:
            vis = vis * np.array((1, 1, 1)).reshape(1, 1, 1, 3)
        vis = tf.image.resize_images(vis, (target_shape[0] - padding * 2, target_shape[1] - padding * 2),
                                     method=tf.image.ResizeMethod.AREA)

        vis = tf.pad(vis,
                     tf.constant([[0, 0], [padding, padding], [padding, padding],
                                  [0, 0]]))
        vis = tf.pad(vis - 1, tf.constant([[0, 0], [4, 4], [4, 4], [0, 0]])) + 1
        visualization[i] = vis

    visualization[3] = visualization[0] * visualization[2]

    visualization_lines = []
    images_per_line = 3
    for i in range(len(visualization) // images_per_line):
        visualization_lines.append(
            tf.concat(
                axis=2,
                values=visualization[i * images_per_line:(i + 1
                                                          ) * images_per_line]))
    visualization = tf.maximum(0.0, tf.concat(axis=1, values=visualization_lines))
    print('visualization shape', visualization.shape)

    return visualization


def get_weighted_variance(image, illums_est):
    confidence = tf.sqrt(tf.reduce_sum(illums_est ** 2, axis=3))
    vis_est = tf.nn.l2_normalize(illums_est, 3)
    ##################
    confidence_dist = confidence[:, :, :, None] / tf.reduce_sum(
        confidence, axis=(1, 2), keep_dims=True)[:, :, :, None]
    mean_est = tf.reduce_mean(vis_est, axis=(1, 2), keep_dims=True)
    sq_deviation = tf.pow(vis_est - mean_est, 2)
    weighted_sq_dev = confidence_dist * sq_deviation
    variance = tf.reduce_sum(weighted_sq_dev, axis=(1, 2))
    ##################
    return variance


def get_gram_matrix(illum_est):
    #    assert illum_est.shape[0] == 1
    width, height = illum_est.get_shape().as_list()[
                        1], illum_est.get_shape().as_list()[2]
    print(illum_est.shape)
    est_points = tf.reshape(illum_est[0], [width * height, 3])
    gram = tf.matmul(tf.transpose(est_points), est_points)
    # todo: we should take the average
    return gram


# draw text on the bottom right corner of an image,
# lines like ['line1', 'line2']
def put_text_on_image(image, lines):
    for i, line in enumerate(lines[::-1]):
        text_width, text_height = cv2.getTextSize(line, cv2.FONT_HERSHEY_TRIPLEX,
                                                  0.4, 1)[0]
        cv2.putText(image, line, (image.shape[1] - text_width,
                                  image.shape[0] - 2 * i * text_height - 10),
                    cv2.FONT_HERSHEY_TRIPLEX, 0.4, [0, 0, 0])

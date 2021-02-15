import os
import pickle
import sys

import cv2
import numpy as np
import tensorflow as tf

import utils
from auxiliary.settings import EPOCHS, OVERRODE, initialize_dataset_config
from auxiliary.utils import load_data
from classes.data.DataProvider import DataProvider
from classes.data.datasets.ChengDataset import ChengDataset
from classes.data.datasets.GehlerDataset import GehlerDataset
from fcn import FCN
from utils import get_session


def get_average(image_packs):
    data = load_data(image_packs.split(','))
    avg = np.zeros(shape=(3,), dtype=np.float32)
    for record in data:
        cv2.imshow('img', cv2.resize((record.img / 2.0 ** 16) ** 0.5, (0, 0), fx=0.2, fy=0.2))
        cv2.waitKey(0)
        avg += np.mean(record.img.astype(np.float32), axis=(0, 1))
    avg = avg / np.linalg.norm(avg)
    print('({:.3f}, {:.3f}, {:.3f})'.format(avg[0], avg[1], avg[2]))


def test(name, ckpt, image_pack_name=None, output_filename=None):
    try:
        external_image = image_pack_name.index('.') != -1
    except:
        external_image = None

    if image_pack_name is None:
        data = None
    elif not external_image:
        print("Loading image pack {}".format(image_pack_name))
        data = load_data(image_pack_name.split(','))

    with get_session() as sess:
        fcn = FCN(sess=sess, name=name)
        if ckpt != "-1":
            fcn.load(ckpt)
        else:
            fcn.load_absolute(name)
        if not external_image:
            errors, _, _, _, ret, conf = fcn.test(scales=[0.5],
                                                  summary=True,
                                                  summary_key=123,
                                                  data=data,
                                                  eval_speed=False,
                                                  visualize=output_filename is None)
            if output_filename is not None:
                try:
                    os.mkdir("outputs")
                except:
                    pass

                pickle.dump(ret, open(os.path.join("outputs", "{}.pkl".format(output_filename)), 'wb'))
                pickle.dump(errors, open(os.path.join("outputs", "{}_err.pkl".format(output_filename)), 'wb'))
                pickle.dump(conf, open(os.path.join("outputs", "{}_conf.pkl".format(output_filename)), 'wb'))
                print(ret)
                print("results dumped to outputs/{}_err.pkl".format(output_filename))
        else:
            img = cv2.imread(image_pack_name)
            # Reverse gamma correction for sRGB
            img = (img / 255.0) ** 2.2 * 65536
            images = [img]
            fcn.test_external(images=images, fns=[image_pack_name])


def dump_result(name, ckpt, image_pack_name=None):
    data = load_data(image_pack_name.split(',')) if image_pack_name is not None else None
    with get_session() as sess:
        fcn = FCN(sess=sess, name=name)
        fcn.load(ckpt)
        _, _, outputs, gts = fcn.test(scales=[0.5], summary=True, summary_key=123, data=data)
    pickle.dump({"outputs": np.array(outputs), "gts": np.array(gts)},
                open(os.path.join("outputs", "{}-{}-{}.pkl".format(name, ckpt, image_pack_name)), "wb"))


def dump_errors(name, ckpt, fold, output_filename, method='full', samples=0, pooling='median'):
    with get_session() as sess:
        fcn = FCN(sess=sess, name=name, kwargs={'dataset_name': 'gehler', 'subset': 0, 'fold': fold})
        fcn.load(ckpt)
        for i in range(4):
            if method == 'full':
                errors, t, _, _, _ = fcn.test(scales=[0.5])
            elif method == 'resize':
                errors, t = fcn.test_resize()
            elif method == 'patches':
                errors, t = fcn.test_patch_based(scale=0.5, patches=samples, pooling=pooling)
            else:
                assert False
    utils.print_angular_errors(errors)
    pickle.dump({'e': errors, 't': t}, open(output_filename, 'w'))


def test_multi(name, ckpt):
    with get_session() as sess:
        fcn = FCN(sess=sess, name=name)
        fcn.load(ckpt)
        fcn.test_multi()


def test_network(name, ckpt):
    with get_session() as sess:
        fcn = FCN(sess=sess, name=name)
        fcn.load(ckpt)
        fcn.test_network()


def cont(name, preload, key):
    with get_session() as sess:
        fcn = FCN(sess=sess, name=name)
        sess.run(tf.global_variables_initializer())
        if preload is not None:
            fcn.load(name=preload, key=key)
        fcn.train(EPOCHS)


def train(name, *args):
    kwargs = {}
    for arg in args:
        key, val = arg.split('=')
        kwargs[key] = val
        OVERRODE[key] = val

    with get_session() as sess:
        fcn = FCN(sess=sess, name=name, kwargs=kwargs)
        sess.run(tf.global_variables_initializer())
        fcn.train(EPOCHS)


def show_patches():
    dp = DataProvider(True, ['g0'])
    dp.set_batch_size(10)
    while True:
        batch = dp.get_batch()
        for img in batch[0]:
            # img = img / np.mean(img, axis=(0, 1))[None, None, :]
            img = img / img.max()
            cv2.imshow("Input", np.power(img, 1 / 2.2))
            cv2.waitKey(0)


def dump_gehler():
    ds = GehlerDataset()
    ds.regenerate_meta_data()
    ds.regenerate_image_packs()


def dump_cheng(start, end):
    for i in range(start, end + 1):
        ds = ChengDataset(i)
        ds.regenerate_meta_data()
        ds.regenerate_image_packs()


def override_global(key, val):
    print("Overriding ", key, '=', val)
    OVERRODE[key] = val
    print(globals()[key])
    initialize_dataset_config()


def test_naive():
    return FCN.test_naive()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: ./fccc.py [func]")
        exit(-1)
    filename = __file__[2:]
    mode = sys.argv[1]
    globals()[mode](*sys.argv[2:])

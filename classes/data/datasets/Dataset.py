import os
import pickle
import sys

import cv2
import numpy as np

from auxiliary.settings import SHOW_IMAGES, FOLDS


class Dataset:

    @staticmethod
    def get_name() -> str:
        pass

    @staticmethod
    def get_subset_name() -> str:
        return ""

    def load_image_without_mcc(self, r):
        pass

    def get_directory(self):
        return os.path.join("data", self.get_name())

    def get_img_directory(self):
        return os.path.join("data", self.get_name())

    def get_metadata_file_name(self):
        return os.path.join(self.get_directory(), self.get_subset_name(), "meta.pkl")

    def dump_meta_data(self, meta_data):
        print("\n Dumping data =>", self.get_metadata_file_name())
        print("\t Total records:", sum(map(len, meta_data)))
        print("\t Slices:", map(len, meta_data))
        pickle.dump(meta_data, open(self.get_metadata_file_name(), 'wb'), protocol=-1)
        print("\n Dumped!")

    def load_meta_data(self):
        return pickle.load(open(self.get_metadata_file_name()))

    def get_image_pack_fn(self, fold):
        return os.path.join(self.get_directory(), self.get_subset_name(), "image_pack.{:d}.pkl".format(fold))

    def dump_image_pack(self, image_pack, fold):
        pickle.dump(image_pack, open(self.get_image_pack_fn(fold), 'wb'), protocol=-1)

    def load_image_pack(self):
        return pickle.load(open(self.get_metadata_file_name()))

    def regenerate_image_pack(self, meta_data, fold):
        image_pack = []
        for i, r in enumerate(meta_data):
            print("Processing {:d}/{:d}\r".format(i + 1, len(meta_data)), end=' ')
            sys.stdout.flush()
            r.img = self.load_image_without_mcc(r)

            if SHOW_IMAGES:
                cv2.imshow("img", cv2.resize(np.power(r.img / 65535., 1.0 / 3.2), (0, 0), fx=0.25, fy=0.25))
                illuminant = r.illuminant
                if len(illuminant.shape) >= 3:
                    cv2.imshow("Illuminant", illuminant)
                cv2.waitKey(0)

            image_pack.append(r)

        self.dump_image_pack(image_pack, fold)

    def regenerate_image_packs(self):
        meta_data = self.load_meta_data()
        print("Dumping image packs...")
        print("{} folds found".format(len(meta_data)))
        for f, m in enumerate(meta_data):
            self.regenerate_image_pack(m, f)

    @staticmethod
    def get_folds():
        return FOLDS

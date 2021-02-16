import os

import cv2
import numpy as np
import scipy.io

from auxiliary.settings import DATA_FRAGMENT, FOLDS, BOARD_FILL_COLOR
from classes.data.ImageRecord import ImageRecord
from classes.data.datasets.Dataset import Dataset


class GehlerDataset(Dataset):

    def get_name(self) -> str:
        return "gehler"

    def regenerate_meta_data(self):
        print("Loading and shuffle fn_and_illum[]")

        meta_data = []
        ground_truth = scipy.io.loadmat(os.path.join(self.get_directory(), "ground_truth.mat"))["real_rgb"]
        ground_truth /= np.linalg.norm(ground_truth, axis=1)[..., np.newaxis]
        filenames = sorted(os.listdir(os.path.join(self.get_directory(), "images")))
        folds = scipy.io.loadmat(os.path.join(self.get_directory(), "folds.mat"))

        for i in range(len(filenames)):
            file_name = filenames[i]
            mcc_coord = self.get_mcc_coord(file_name)
            meta_data.append(ImageRecord(self.get_name(), file_name, ground_truth[i], mcc_coord))

        if DATA_FRAGMENT != -1:
            meta_data = meta_data[:DATA_FRAGMENT]
            print("Warning: using only first {:d} images...".format(len(meta_data)))

        meta_data_folds = [[], [], []]
        for i in range(FOLDS):
            fold = list(folds['te_split'][0][i][0])
            print(len(fold))
            for j in fold:
                meta_data_folds[i].append(meta_data[j - 1])

        for i in range(3):
            print('Fold', i)
            print(map(lambda m: m.file_name, meta_data_folds[i]))
        print(sum(map(len, meta_data_folds)))
        assert sum(map(len, meta_data_folds)) == len(filenames)

        for i in range(3):
            assert set(meta_data_folds[i]) & set(meta_data_folds[(i + 1) % 3]) == set()

        self.dump_meta_data(meta_data_folds)

    def get_mcc_coord(self, file_name):
        path_to_coords = os.path.join(self.get_directory(), "coordinates", file_name.split('.')[0] + '_macbeth.txt')
        lines = open(path_to_coords, 'r').readlines()
        width, height = map(float, lines[0].split())
        scale_x, scale_y = 1 / width, 1 / height
        lines = [lines[1], lines[2], lines[4], lines[3]]
        polygon = []
        for line in lines:
            line = line.strip().split()
            x, y = (scale_x * float(line[0])), (scale_y * float(line[1]))
            polygon.append((x, y))
        return np.array(polygon, dtype="float32")

    def load_image(self, file_name: str) -> np.array:
        file_path = self.get_img_directory() + '/images/' + file_name
        raw = np.array(cv2.imread(file_path, -1), dtype='float32')
        black_point = 129 if file_name.startswith('IMG') else 1
        raw = np.maximum(raw - black_point, [0, 0, 0])
        return raw

    def load_image_without_mcc(self, r) -> np.array:
        raw = self.load_image(r.file_name)
        img = (np.clip(raw / raw.max(), 0, 1) * 65535.0).astype(np.uint16)
        polygon = r.mcc_coord * np.array([img.shape[1], img.shape[0]])
        polygon = polygon.astype(np.int32)
        cv2.fillPoly(img, [polygon], (BOARD_FILL_COLOR,) * 3)
        return img

import os

# --------------------------------
#       Dataset
# --------------------------------

DATASET_NAME = 'gehler'
SUBSET = 0
FOLDS = 3
FOLD = 0
TRAINING_FOLDS, TEST_FOLDS = [], []
SHOW_IMAGES = False
DATA_FRAGMENT = -1
BOARD_FILL_COLOR = 1e-5

# --------------------------------
#       Model
# --------------------------------

PATH_TO_MODEL = os.path.join("models", "fc4")
FCN_INPUT_SIZE = 512


def initialize_dataset_config(dataset_name=None, subset=None):
    global DATASET_NAME, SUBSET, FOLD
    if dataset_name is not None:
        DATASET_NAME = dataset_name
        SUBSET = subset

    global TRAINING_FOLDS, TEST_FOLDS
    if DATASET_NAME == "gehler":
        T = FOLD
        print("FOLD", FOLD)
        if T != -1:
            TRAINING_FOLDS = ['g{:d}'.format(T), 'g{:d}'.format((T + 1) % 3)]
            TEST_FOLDS = ['g{:d}'.format((T + 2) % 3)]
        else:
            TRAINING_FOLDS = []
            TEST_FOLDS = ['g0', 'g1', 'g2']
    elif DATASET_NAME == "cheng":
        subset = SUBSET
        T = FOLD
        TRAINING_FOLDS = ["c{}{:d}".format(subset, T), "c{}{:d}".format(subset, (T + 1) % 3)]
        TEST_FOLDS = ['c{}{:d}'.format(subset, (T + 2) % 3)]
    elif DATASET_NAME == 'multi':
        TEST_FOLDS = ['multi']

    print(TRAINING_FOLDS)
    print(TEST_FOLDS)
    return TRAINING_FOLDS, TEST_FOLDS


OVERRODE = {}

# --------------------------------
#       Dataset
# --------------------------------

SHOW_IMAGES = False
FOLDS = 3
DATA_FRAGMENT = -1
BOARD_FILL_COLOR = 1e-5

# --------------------------------
#       Model
# --------------------------------

PATH_TO_MODEL = "models/fc4"
FCN_INPUT_SIZE = 512

# --------------------------------
#       Data augmentation
# --------------------------------

# Use data augmentation?
AUGMENTATION = True

# Rotation angle
AUGMENTATION_ANGLE = 60

# Patch scale
AUGMENTATION_SCALE = [0.1, 1.0]

# Random left-right flip?
AUGMENTATION_FLIP_LEFTRIGHT = True

# Random top-down flip?
AUGMENTATION_FLIP_TOPDOWN = False

# Color rescaling?
AUGMENTATION_COLOR = 0.8

# Cross-channel terms
AUGMENTATION_COLOR_OFFDIAG = 0.0

# Augment Gamma?
AUGMENTATION_GAMMA = 0.0

# Augment using a polynomial curve?
USE_CURVE = False

# Apply different gamma and curve to left/right halves?
SPATIALLY_VARIANT = False

# The gamma used in the AlexNet branch to make patches in sRGB
INPUT_GAMMA = 2.2

# The gamma for visualization
VIS_GAMMA = 2.2

# Shuffle the images, after each epoch?
DATA_SHUFFLE = True


# Data Sets
DATASET_NAME = 'gehler'
SUBSET = 0
FOLD = 0
TRAINING_FOLDS = []
TEST_FOLDS = []


def initialize_dataset_config(dataset_name=None, subset=None, fold=None):
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



# --------------------------------
#       Test
# --------------------------------




OVERRODE = {}

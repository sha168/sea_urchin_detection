import torch
from pathlib import Path


BATCH_SIZE = 10  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # resize the image for training and transforms
NUM_EPOCHS = 100  # number of epochs to train for
NUM_WORKERS = 4
LR = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
DATASET = 'UIQS'  # 'AUDD' or 'UIQS'

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = 'results_audd'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '/experiment/' + DATASET + '/train'
VALID_DIR = '/experiment/' + DATASET + '/test'

# TRAIN_DIR = '/Users/sha168/Downloads/' + DATASET + '/train'
# VALID_DIR = '/Users/sha168/Downloads/' + DATASET + '/test'

# classes: 0 index is reserved for background
if DATASET == 'UIQS':
    CLASSES = [
        'background', 'urchin', 'holothurian', 'scallop'
    ]
elif DATASET == 'AUDD':
    CLASSES = [
        'background', 'seacucumber', 'seaurchin', 'scallop'
    ]

NUM_CLASSES = len(CLASSES)

# location to save model and plots
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 50  # save model after these many epochs
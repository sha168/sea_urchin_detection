import torch
from pathlib import Path
import numpy as np

BATCH_SIZE = 10  # increase / decrease according to GPU memeory
RESIZE_TO = 512  # 512  # resize the image for training and transforms
NUM_EPOCHS = 3  # number of epochs to train for
NUM_WORKERS = 0
LR = 0.001
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
DATASET = 'TROMSO'  # 'AUDD' or 'UIQS'

# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False

OUT_DIR = 'results_tromso'
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# TRAIN_DIR = '/experiment/' + DATASET + '/train'
# VALID_DIR = '/experiment/' + DATASET + '/test'

TRAIN_DIR = 'gdrive/MyDrive/sea_urchin_data/' + DATASET + '/train'
VALID_DIR = 'gdrive/MyDrive/sea_urchin_data/' + DATASET + '/test'

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
elif DATASET == 'TROMSO':
    CLASSES = [
        'background', 'urchin'
    ]

NUM_CLASSES_PT = 4
NUM_CLASSES = len(CLASSES)
COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

# location to save model and plots
SAVE_PLOTS_EPOCH = 2  # save loss plots after these many epochs
SAVE_MODEL_EPOCH = 1  # save model after these many epochs

# Testing video
PRETRAINED = 'gdrive/MyDrive/sea_urchin_data/pretrained_models/audd_model10.pth'
VIDEO_IN = 'gdrive/MyDrive/sea_urchin_data/videos/video_2023-02-20_resized.MP4'
VIDEO_OUT = 'detected.mp4'
PERIOD = 2
PROB_THRES = 0.5



from engine import train_one_epoch, evaluate
import utils
import torch
from dataset import CocoDataset

TRAIN = 'experiments/AUDD/images/train'
TEST = 'experiments/AUDD/images/test'
TRAIN_ANN = 'experiments/AUDD/annotations/instances_train.json'
TEST_ANN = 'experiments/AUDD/annotations/instances_test.json'

# TRAIN = '/Users/sha168/Downloads/AUDD/images/train'
# TEST = '/Users/sha168/Downloads/AUDD/images/test'
# TRAIN_ANN = '/Users/sha168/Downloads/AUDD/annotations/instances_train.json'
# TEST_ANN = '/Users/sha168/Downloads/AUDD/annotations/instances_test.json'

N_CLASS = 4
LR = 0.005
WEIGHT_DECAY = 0.0005
MOMENTUM = 0.9
NUM_EPOCHS = 10
BATCH_SIZE = 10

def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = N_CLASS
    # use our dataset and defined transformations
    dataset = CocoDataset(data_dir=TRAIN, anno_file_path=TRAIN_ANN, transforms=utils.get_transform(train=True))
    dataset_test = CocoDataset(data_dir=TEST, anno_file_path=TEST_ANN, transforms=utils.get_transform(train=False))

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = utils.get_model(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=LR,
                                momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for NUM_EPOCHS epochs
    num_epochs = NUM_EPOCHS

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("Training is done!")


if __name__ == '__main__':
    main()

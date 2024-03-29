from sea_urchin_detection.config import DEVICE, NUM_CLASSES, NUM_CLASSES_PT, NUM_EPOCHS, OUT_DIR
from sea_urchin_detection.config import VISUALIZE_TRANSFORMED_IMAGES
from sea_urchin_detection.config import SAVE_PLOTS_EPOCH, SAVE_MODEL_EPOCH
from sea_urchin_detection.config import LR, WEIGHT_DECAY, MOMENTUM, PRETRAINED
from sea_urchin_detection.model import create_model
from sea_urchin_detection.utils import Averager
from tqdm import tqdm
from sea_urchin_detection.datasets import train_loader, valid_loader
import torch
import matplotlib.pyplot as plt
import time
from sea_urchin_detection.utils import evaluate, draw_boxes
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
plt.style.use('ggplot')

val_itr = 1
val_loss_list = []
train_itr = 1
train_loss_list = []

# function for running training iterations
def train(train_data_loader, model, optimizer, train_loss_hist):
    global train_itr
    global train_loss_list
    print('Training')

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))

    model.train()
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)
        train_loss_hist.send(loss_value)
        losses.backward()
        optimizer.step()
        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return train_loss_list


# function for running validation iterations
def validate(valid_data_loader, model, val_loss_hist):
    print('Validating')
    global val_itr
    global val_loss_list

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    # model.eval()
    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list


def predict(data_loader, model, prob_thresh):
    print('Validating')

    # initialize tqdm progress bar
    #prog_bar = tqdm(data_loader, total=len(data_loader))

    model.eval()
    for i, data in enumerate(data_loader):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)

        with torch.no_grad():
            predictions = model(images)


        break

    for i, image in enumerate(images):
        prediction = predictions[i]
        boxes_ = prediction['boxes'].cpu().numpy()
        labels_ = prediction['labels'].cpu().numpy()
        probs_ = prediction['scores'].cpu().numpy()

        kept_indices = probs_ > prob_thresh
        boxes_ = boxes_[kept_indices]
        labels_ = labels_[kept_indices]

        image_ = image.permute(1, 2, 0).cpu().numpy()

        draw_boxes(boxes_, labels_, image_, i)


def fine_tune(lr, epochs, prob_thresh):

    # initialize the model and move to the computation device
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    # get the number of input features
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES_PT)

    if PRETRAINED != None:
        model.load_state_dict(torch.load(PRETRAINED, map_location="cpu"))

    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)

    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    optimizer = torch.optim.SGD(params, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_precision_hist = Averager()
    val_precision_hist = Averager()
    train_recall_hist = Averager()
    val_recall_hist = Averager()

    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations

    train_precision_list = []
    val_precision_list = []
    train_recall_list = []
    val_recall_list = []
    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from utils import show_tranformed_image

        show_tranformed_image(train_loader)
    # start the training epochs
    for epoch in range(epochs):
        print(f"\nEPOCH {epoch + 1} of {epochs}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        train_precision_hist.reset()
        val_precision_hist.reset()
        train_recall_hist.reset()
        val_recall_hist.reset()
        # create two subplots, one for each, loss and precision/recall
        figure_1, train_ax = plt.subplots()
        figure_2, valid_ax = plt.subplots()
        # start timer and carry out training and validation
        start = time.time()
        # predict(valid_loader, model)
        train_loss = train(train_loader, model, optimizer, train_loss_hist)
        if epoch % 20 == 0:
            val_loss = validate(valid_loader, model, val_loss_hist)
            train_evaluator = evaluate(model, train_loader, DEVICE)
            val_evaluator = evaluate(model, valid_loader, DEVICE)

            train_stats = train_evaluator.coco_eval['bbox'].stats.__array__()
            val_stats = val_evaluator.coco_eval['bbox'].stats.__array__()

            train_precision_list.append(train_stats[1])
            train_precision_hist.send(train_stats[1])
            train_recall_list.append(train_stats[8])
            train_recall_hist.send(train_stats[8])
            val_precision_list.append(val_stats[1])
            val_precision_hist.send(val_stats[1])
            val_recall_list.append(val_stats[8])
            val_recall_hist.send(val_stats[8])

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:
            predict(valid_loader, model, prob_thresh)

        if (epoch + 1) == epochs:  # save loss plots and model once at the end
            torch.save(model.state_dict(), f"{OUT_DIR}/model.pth")

        # update the learning rate
        lr_scheduler.step()
        print(f"Epoch #{epoch +1} train loss: {train_loss_hist.value:.3f}")
        #print(f"Epoch #{epoch +1} validation precision: {val_precision_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # if (epoch + 1) % SAVE_MODEL_EPOCH == 0:  # save model after every n epochs
        #     torch.save(model.state_dict(), f"{OUT_DIR}/model{epoch + 1}.pth")
        #     print('SAVING MODEL COMPLETE...\n')

        if (epoch + 1) % SAVE_PLOTS_EPOCH == 0:  # save loss plots after n epochs
            train_ax.plot(train_loss, color='blue', label='train')
            train_ax.plot(val_loss, color='red', label='val')
            train_ax.legend()
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('loss')
            valid_ax.plot(val_precision_list, color='red', label='val precision')
            valid_ax.plot(train_precision_list, color='blue', label='train precision')
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('precision')
            valid_ax.legend()
            valid_ax2 = valid_ax.twinx()
            valid_ax2.plot(val_recall_list, color='red', linestyle='--', label='val recall')
            valid_ax2.plot(train_recall_list, color='blue', linestyle='--', label='train recall')
            valid_ax2.set_ylabel('recall')
            valid_ax2.legend(loc='lower right')
            figure_1.savefig(f"{OUT_DIR}/loss.png")  # _{epoch + 1}
            figure_2.savefig(f"{OUT_DIR}/precision.png")  # _{epoch + 1}
            print('SAVING PLOTS COMPLETE...')

        if (epoch + 0) == epochs:  # save loss plots and model once at the end
            train_ax.plot(train_loss, color='blue', label='train')
            train_ax.plot(val_loss, color='red', label='val')
            train_ax.legend()
            train_ax.set_xlabel('iterations')
            train_ax.set_ylabel('loss')
            valid_ax.plot(val_precision_list, color='red', label='val')
            valid_ax.plot(train_precision_list, color='blue', label='train')
            valid_ax.legend()
            valid_ax.set_xlabel('iterations')
            valid_ax.set_ylabel('precision')
            valid_ax2 = valid_ax.twinx()
            valid_ax2.plot(val_recall_list, color='red', linestyle='--', label='val recall')
            valid_ax2.plot(train_recall_list, color='blue', linestyle='--', label='train recall')
            valid_ax2.set_ylabel('recall')
            valid_ax2.legend(loc='lower right')
            figure_1.savefig(f"{OUT_DIR}/loss_{epoch + 1}.png")
            figure_2.savefig(f"{OUT_DIR}/precision_{epoch + 1}.png")
            torch.save(model.state_dict(), f"{OUT_DIR}/model.pth")

        plt.close('all')
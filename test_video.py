import itertools
import random
import time
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from model import create_model
from torchvision.transforms import transforms

from config import NUM_CLASSES, PRETRAINED, DEVICE, RESIZE_TO, PERIOD, PROB_THRES, VIDEO
from bbox import BBox


def _infer_stream(path_to_input_stream_endpoint, period_of_inference, prob_thresh):

    model = create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(DEVICE)
    model.eval()

    if path_to_input_stream_endpoint.isdigit():
        path_to_input_stream_endpoint = int(path_to_input_stream_endpoint)
    video_capture = cv2.VideoCapture(path_to_input_stream_endpoint)

    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (512, 1024))

    with torch.no_grad():
        for sn in itertools.count(start=1):
            _, frame = video_capture.read()

            if sn % period_of_inference != 0:
                continue

            timestamp = time.time()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)

            # scale = RESIZE_TO / min(image.height, image.width)

            # image_resized = transforms.Resize((round(image.height * scale), round(image.width * scale)))(image)
            image_resized = image
            image_tensor = transforms.ToTensor()(image_resized).to(DEVICE)

            predictions = model(image_tensor.unsqueeze(dim=0))
            detection_bboxes = predictions[0]['boxes']
            detection_classes = predictions[0]['labels']
            detection_probs = predictions[0]['scores']

            # detection_bboxes /= scale

            kept_indices = detection_probs > prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]

            fig, ax = plt.subplots()

            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                if cls == 1:  # only interested in urchins
                    color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                    #bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                    category = 'urchin'

                    #draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                    #draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

                    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1,
                                             edgecolor=color, facecolor='none')
                    ax.add_patch(rect)
                    text = plt.text(bbox[2], bbox[3], s=f'{category:s} {prob:.3f}', color=color)

                    # Add the patch to the Axes
                    ax.add_patch(rect)


            elapse = time.time() - timestamp
            fps = 1 / elapse

            plt.text(0, 0, s=f'FPS = {fps:.1f}', color='r')
            plt.show()
            #
            # if cv2.waitKey(10) == 27:
            #     break

            # out.write(frame)
            # # cv2.imshow('easy-faster-rcnn.pytorch', frame)
            # c = cv2.waitKey(1)
            # if c & 0xFF == ord('q'):
            #     break

    video_capture.release()
    #out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    def main():

        _infer_stream(VIDEO, PERIOD, PROB_THRES)

    main()
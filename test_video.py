import itertools
import random
import time
import torch
import cv2
import numpy as np
from PIL import ImageDraw, Image
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

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

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

            draw = ImageDraw.Draw(image)

            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                if cls == 1:  # only interested in urchins
                    color = random.choice(['red', 'green', 'blue', 'yellow', 'purple', 'white'])
                    bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                    category = 'urchin'

                    draw.rectangle(((bbox.left, bbox.top), (bbox.right, bbox.bottom)), outline=color)
                    draw.text((bbox.left, bbox.top), text=f'{category:s} {prob:.3f}', fill=color)

            image = np.array(image)
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            elapse = time.time() - timestamp
            fps = 1 / elapse
            cv2.putText(frame, f'FPS = {fps:.1f}', (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # cv2.imshow('easy-faster-rcnn.pytorch', frame)
            # if cv2.waitKey(10) == 27:
            #     break

            out.write(frame)
            c = cv2.waitKey(1)
            if c & 0xFF == ord('q'):
                break

    video_capture.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    def main():

        _infer_stream(VIDEO, PERIOD, PROB_THRES)

    main()
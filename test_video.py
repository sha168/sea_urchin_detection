import itertools
import random
import time
import torch
import cv2
from PIL import Image
from model import create_model
from torchvision.transforms import transforms
import numpy as np
from config import NUM_CLASSES, PRETRAINED, DEVICE, RESIZE_TO, PERIOD, PROB_THRES, VIDEO_IN, VIDEO_OUT
from bbox import BBox

def _infer_stream(path_to_input_stream_endpoint, path_to_output_stream_endpoint, period_of_inference, prob_thresh):

    model = create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(DEVICE)
    model.eval()

    # Initialize the video stream and pointer to output video file
    vs = cv2.VideoCapture(path_to_input_stream_endpoint)
    writer = None
    vs.set(cv2.CAP_PROP_POS_FRAMES, 1000);

    with torch.no_grad():
        for sn in itertools.count(start=1):

            (grabbed, frame) = vs.read()

            if sn % period_of_inference != 0:
                continue

            image_tensor = transforms.ToTensor()(frame).to(DEVICE)

            predictions = model(image_tensor.unsqueeze(dim=0))
            detection_bboxes = predictions[0]['boxes']
            detection_classes = predictions[0]['labels']
            detection_probs = predictions[0]['scores']

            kept_indices = detection_probs > prob_thresh
            detection_bboxes = detection_bboxes[kept_indices]
            detection_classes = detection_classes[kept_indices]
            detection_probs = detection_probs[kept_indices]

            masked_frame = frame
            for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(), detection_probs.tolist()):
                if cls == 1:  # only interested in urchins
                    color = list(np.random.random(size=3) * 256)
                    bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                    category = 'urchin'

                    masked_frame = cv2.rectangle(masked_frame, (int(bbox.left), int(bbox.top)), (int(bbox.right), int(bbox.bottom)), color, 2)
                    masked_frame = cv2.putText(
                        masked_frame, (int(bbox.left), int(bbox.top)), f'{category:s} {prob:.3f}', cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
                    )

            # Check if the video writer is None
            if writer is None:
                # Initialize our video writer
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(path_to_output_stream_endpoint, fourcc, 10,
                                         (masked_frame.shape[1], masked_frame.shape[0]), True)

            # Write the output frame to disk
            writer.write(masked_frame)

        # Release the file pointers
        print("[INFO] cleaning up...")
        writer.release()


if __name__ == '__main__':
    def main():

        _infer_stream(VIDEO_IN, VIDEO_OUT, PERIOD, PROB_THRES)

    main()



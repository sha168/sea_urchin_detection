import torch
import cv2
from sea_urchin_detection.model import create_model
from torchvision.transforms import transforms
import numpy as np
from sea_urchin_detection.config import NUM_CLASSES, PRETRAINED, DEVICE, PERIOD, PROB_THRES, VIDEO_IN, VIDEO_OUT
from sea_urchin_detection.bbox import BBox

def test_video(path_to_input_stream_endpoint, prob_thresh):

    path_to_input_stream_endpoint = 'gdrive/MyDrive/sea_urchin_data/videos/' + path_to_input_stream_endpoint
    path_to_output_stream_endpoint = 'detected_pretrained.mp4'
    period_of_inference = 2

    model = create_model(num_classes=NUM_CLASSES, pretrained=PRETRAINED)
    model = model.to(DEVICE)
    model.eval()

    # Initialize the video stream and pointer to output video file
    vs = cv2.VideoCapture(path_to_input_stream_endpoint)
    # vs.set(cv2.CAP_PROP_POS_FRAMES, 7000)

    if vs.isOpened() == False:
        print("Error reading video file")

    # We need to set resolutions.
    # so, convert them from float to integer.
    frame_width = int(vs.get(3))
    frame_height = int(vs.get(4))

    size = (frame_width, frame_height)

    # Initialize our video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(path_to_output_stream_endpoint, fourcc, 5,
                             size, True)

    i_f = 0
    with torch.no_grad():
        while(True):

            grabbed, frame = vs.read()
            i_f += 1

            if i_f % period_of_inference != 0:
                continue

            if grabbed:

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
                for bbox, cls, prob in zip(detection_bboxes.tolist(), detection_classes.tolist(),
                                           detection_probs.tolist()):
                    if cls == 2:  # only interested in urchins
                        color = list(np.random.random(size=3) * 256)
                        color = [256, 0, 0]
                        bbox = BBox(left=bbox[0], top=bbox[1], right=bbox[2], bottom=bbox[3])
                        category = 'urchin'

                        masked_frame = cv2.rectangle(masked_frame, (int(bbox.left), int(bbox.top)),
                                                     (int(bbox.right), int(bbox.bottom)), color, 2)
                        masked_frame = cv2.putText(
                            masked_frame, f'{category:s} {prob:.3f}', (int(bbox.left), int(bbox.top)),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
                        )

                if len(detection_bboxes.tolist()) > 0:
                    print('Detected ' + str(len(detection_bboxes.tolist())) +
                          ' sea urchin(s) in frame #' + str(i_f) + '!', flush=True)

                # Write the output frame to disk
                writer.write(masked_frame)

            # Break the loop
            else:
                break

    # Release the file pointers
    print("Cleaning up...", flush=True)
    vs.release()
    writer.release()

    # Closes all the frames
    cv2.destroyAllWindows()

    print("The video was successfully saved!", flush=True)


if __name__ == '__main__':
    def main():

        test_video(VIDEO_IN, PROB_THRES)

    main()



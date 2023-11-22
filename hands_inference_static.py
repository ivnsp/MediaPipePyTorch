import numpy as np
import torch
import cv2
import sys
import os
import glob


from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_roi


here = os.path.dirname(os.path.abspath(__file__))
PATH_TO_HAND_DETECTION_MODEL = os.path.join(here, "blazepalm.pth")
PATH_TO_SSD_ANCHORS = os.path.join(here, "anchors_palm.npy")
PATH_TO_HAND_LANDMARK_MODEL = os.path.join(here, "blazehand_landmark.pth")


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(gpu))
torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(PATH_TO_HAND_DETECTION_MODEL)
palm_detector.load_anchors(PATH_TO_SSD_ANCHORS)
palm_detector.min_score_thresh = .5

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights(PATH_TO_HAND_LANDMARK_MODEL)

print("Models loaded successfully.")

def model_inference(frame):
    """
    apply inference from blazepalm and blazehand_landmark models
    frame: numpy array of shape (height, width, RGB)
    """

    # TODO: introduce resize_pad_[size] function for specific needs only
    img, _, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
    #img, _, box = hand_regressor.extract_roi(frame, xc, yc, theta, scale)

    return palm_detections#, box


def main(img_dir, show=True, save_dir=None):
    """Main function.
    Args:
        img_dir: path to the directory containing the images to be processed.
    """

    with open('pytorch__detection_stats.txt', 'w') as f:
        for i, imgpath in enumerate(os.listdir(img_dir)):
            img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, imgpath)), cv2.COLOR_BGR2RGB)
            detections = model_inference(img)
            #detections, box = model_inference(img)
            draw_detections(img, detections)
            #draw_roi(img, box)
            
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if show:
                cv2.imshow(f'Detections in {os.path.basename(imgpath)}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, os.path.basename(imgpath)), img)

            print("File: ", os.path.basename(imgpath), '\t', len(detections), " hands detected")
            f.write(f'File: {os.path.basename(imgpath)}\t{len(detections)} hands detected\n')
            f.write(str(detections) + '\n')
            #f.write(str(box) + '\n')

    return


if __name__ == '__main__':

    import sys

    main(sys.argv[1], sys.argv[2], sys.argv[3])


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


PATH_TO_HAND_DETECTION_MODEL = "/Users/rach3project/Repos/MediaPipePyTorch/blazepalm.pth"
PATH_TO_SSD_ANCHORS = "/Users/rach3project/Repos/MediaPipePyTorch/anchors_palm.npy"
PATH_TO_HAND_LANDMARK_MODEL = "/Users/rach3project/Repos/MediaPipePyTorch/blazehand_landmark.pth"


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(gpu))
torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(PATH_TO_HAND_DETECTION_MODEL)
palm_detector.load_anchors(PATH_TO_SSD_ANCHORS)
palm_detector.min_score_thresh = .75

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
    img, _, box = hand_regressor.extract_roi(frame, xc, yc, theta, scale)

    return palm_detections, box


def main(img_dir, show=True, save_dir=None):
    """Main function.
    Args:
        img_dir: path to the directory containing the images to be processed.
    """
        
    img_list = []
    img_paths = glob.glob(img_dir + '/*.jpg')
    for path in img_paths:
        img_list.append(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB))

    with open('pytorch__detection_stats.txt', 'w') as f:
        for i, img in enumerate(img_list):
            cv2.imshow('img', img)
            cv2.waitKey(0)
            detections, box = model_inference(img)
            print(detections)
            draw_detections(img, detections)

            draw_roi(img, box)
            
            img_list[i] = img

            if show:
                cv2.imshow(f'Detections in {os.path.basename(img_paths[i])}', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, os.path.basename(img_paths[i])), img)

            print("File: ", os.path.basename(img_paths[i]), '\t', len(detections), " hands detected")
            f.write(f'File: {os.path.basename(img_paths[i])}\t{len(detections)} hands detected\n')

    return


if __name__ == '__main__':

    import sys

    main(sys.argv[1], show=True, save_dir=None)


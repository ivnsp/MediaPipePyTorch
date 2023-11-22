import numpy as np
import torch
import cv2
import sys
import os

from PIL import Image
from subprocess import Popen, PIPE

from blazebase import resize_pad, denormalize_detections
from blazepalm import BlazePalm
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS

here = os.path.dirname(os.path.abspath(__file__))
PATH_TO_HAND_DETECTION_MODEL = os.path.join(here, "blazepalm.pth")
PATH_TO_SSD_ANCHORS = os.path.join(here, "anchors_palm.npy")
#PATH_TO_HAND_LANDMARK_MODEL = os.path.join(here, "blazehand_landmark.pth")

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(PATH_TO_HAND_DETECTION_MODEL)
palm_detector.load_anchors(PATH_TO_SSD_ANCHORS)
palm_detector.min_score_thresh = .5

#hand_regressor = BlazeHandLandmark().to(gpu)
#hand_regressor.load_weights(PATH_TO_HAND_LANDMARK_MODEL)


def palm_detection(frame):
    """
    apply inference from blazepalm model to detect palms
    frame: numpy array of shape (height, width, RGB)
    """

    # TODO: introduce resize_pad_[size] function for specific needs only
    img, _, scale, pad = resize_pad(frame)

    normalized_palm_detections = palm_detector.predict_on_image(img)
    palm_detections = denormalize_detections(normalized_palm_detections, scale, pad)

    xc, yc, scale, theta = palm_detector.detection2roi(palm_detections.cpu())
    #img, affine, box = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    #flags, handed, normalized_landmarks = hand_regressor(img.to(gpu))
    #landmarks = hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)

    return palm_detections, xc, yc, scale, theta


def main(inp_dir, out_dir, save_output=True, inference_mode='Image'):
    """
    Main function for inference
    inp_dir: path to the directory containing the images/videos to be processed.
    output_dir: path to the directory where the output files will be saved.
    save_output: whether to save the images with bounding boxes drawn on them.
    model_inference_mode: whether to process images independently (Image), 
    or using add tracking information (Video), or video as independent frames(VideoStatic).
    """
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if inference_mode == 'Image':
        print("Processing images in directory: {}".format(inp_dir))
        for f in os.listdir(inp_dir):
            if f.endswith(".jpg") or f.endswith(".png"):

                img = cv2.imread(os.path.join(inp_dir, f))
                palm_detections, xc, yc, scale, theta = palm_detection(img)
                

                #img = draw_roi(img, xc, yc, scale, theta)
                img = draw_detections(img, palm_detections)
                if save_output:
                    cv2.imwrite(os.path.join(out_dir, f), img)
                #print(palm_detections)
                print("Processed file: {}".format(f), end='\r')

    '''elif inference_mode == 'VideoStatic':
        print("Processing videos in directory: {}".format(inp_dir))
        for f in os.listdir(inp_dir):
            if f.endswith(".mp4") or f.endswith(".avi"):
                vidcap = cv2.VideoCapture(os.path.join(inp_dir, f))
                success, img = vidcap.read()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(os.path.join(out_dir, f), fourcc, 20.0, (img.shape[1], img.shape[0]))
                count = 0
                while success:
                    palm_detections, xc, yc, scale, theta = palm_detection(img)
                    #img = draw_roi(img, xc, yc, scale, theta)
                    img = draw_detections(img, palm_detections)
                    if save_output:
                        # add frame to stream and write video to output file
                        video.write(img)
                    success, img = vidcap.read()
                    print("Processed file: {}".format(f), end='\r')
                    count += 1
                video.release()
                print("Processed file: {}".format(f))
                print("Total frames processed: {}".format(count))

    elif inference_mode == 'Video':
        pass'''


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='Test Palm Detector of MediaPipePytorch')
    parser.add_argument('--inp_dir', type=str, help='input directory')
    parser.add_argument('--out_dir', type=str, help='output directory')
    parser.add_argument('--save_output', type=bool, default=True, help='whether to save output')
    parser.add_argument('--inference_mode', type=str, default='Image', help='Image, Video (not yet implemented), VideoStatic')
    args = parser.parse_args()

    main(args.inp_dir, args.out_dir, args.save_output, args.inference_mode)


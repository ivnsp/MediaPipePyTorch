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


PATH_TO_HAND_DETECTION_MODEL = "/Users/rach3project/Repos/MediaPipePyTorch/blazepalm.pth"
PATH_TO_SSD_ANCHORS = "/Users/rach3project/Repos/MediaPipePyTorch/anchors_palm.npy"
PATH_TO_HAND_LANDMARK_MODEL = "/Users/rach3project/Repos/MediaPipePyTorch/blazehand_landmark.pth"


gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(PATH_TO_HAND_DETECTION_MODEL)
palm_detector.load_anchors(PATH_TO_SSD_ANCHORS)
palm_detector.min_score_thresh = .75

hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights(PATH_TO_HAND_LANDMARK_MODEL)


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
    img, affine, box = hand_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, handed, normalized_landmarks = hand_regressor(img.to(gpu))
    landmarks = hand_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)

    return palm_detections, landmarks, handed, flags, box


def main(inpvid, outdir, save_video=False):

    vidfn = os.path.basename(inpvid).split('.')[0]
    output = os.path.join(outdir, '{}-ld-pytorch.txt'.format(vidfn))
    outarr = os.path.join(outdir, '{}-ld-pytorch.npy'.format(vidfn))

    cap = cv2.VideoCapture(inpvid)
    if save_video:
        p = Popen(['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'mjpeg', '-r', '60',
                '-i', '-', '-vcodec', 'h264', '-qscale', '5', '-r', '60', '{}-ptdraw.mp4'.format(vidfn)],
                stdin=PIPE)

    with open(output, 'w') as outf:

        frame_num, detected = 0, 0

        # init array to append detections to - TODO: remove empty row from beginning
        det_arr = np.zeros((1, 2, 21, 6))

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                outf.write('Detection ratio: {}'.format(detected/(frame_num+1)))
                break

            # Mark the image as not writeable to pass by reference
            image.flags.writeable = False
            
            # init empty detection array - TODO: see above
            #detection = np.zeros((1, 2, 21, 6))
            
            # run both models on frame
            image = np.ascontiguousarray(image[:,:,::-1])
            results = model_inference(image)
            palm_detections, landmarks, handed, flags, box = results

            if flags is not None:
                detected += 1

                for i in range(len(flags)):
                    landmark, flag = landmarks[i], flags[i]
                    if flag>.5 and save_video:
                        image.flags.writeable = True
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        draw_landmarks(image, landmark[:,:2], HAND_CONNECTIONS, size=2)
                        draw_roi(image, box)
                        draw_detections(image, palm_detections)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        im = Image.fromarray(image)
                        im.save(p.stdin, 'JPEG')

                
                # TODO: check for duplicate detections and take one with higher score
                pass
                

                frame_num += 1
                print('Processed {} frames...'.format(frame_num))
                if cv2.waitKey(5) & 0xFF == 27:
                    break

        if save_video:
            p.stdin.close()
            p.wait()
        cap.release()
        cv2.destroyAllWindows()

        # TODO: remove empty row from beginning

        np.save(outarr, det_arr)

        return None


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(description='MediaPipePyTorch hand detection and landmarking with optional output of video with drawings.')
    parser.add_argument('--inpvid', '-i',
                        help='Path to input video file',
                        default="",
                        type=str)
    parser.add_argument('--outdir', '-o',
                        help='Output file directory',
                        type=str,
                        default=".")
    parser.add_argument('--save_video', '-s',
                        help='Whether to save video with drawings as a new file',
                        action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    main(**vars(args))
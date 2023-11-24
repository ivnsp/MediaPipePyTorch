import torch
import os

from blazehand_landmark import BlazeHandLandmark
from blazepalm import BlazePalm


here = os.path.dirname(os.path.abspath(__file__))
PATH_TO_HAND_DETECTION_MODEL = os.path.join(here, "blazepalm.pth")
PATH_TO_SSD_ANCHORS = os.path.join(here, "anchors_palm.npy")
PATH_TO_HAND_LANDMARK_MODEL = os.path.join(here, "blazehand_landmark.pth")

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.set_grad_enabled(False)

palm_detector = BlazePalm().to(gpu)
palm_detector.load_weights(PATH_TO_HAND_DETECTION_MODEL)
palm_detector.load_anchors(PATH_TO_SSD_ANCHORS)
hand_regressor = BlazeHandLandmark().to(gpu)
hand_regressor.load_weights(PATH_TO_HAND_LANDMARK_MODEL)
print("Models loaded successfully.")

print("Checking Palm Detector for BatchNorm2d layers...")
count = 0
for m in palm_detector.modules():
        #print(m)
        if isinstance(m, torch.nn.BatchNorm2d):
            count += 1
            m.track_running_stats = False
            m.eval()
            print("BatchNorm2d layer {} is freezed".format(count))
print("Number of BatchNorm2d layers freezed: {}".format(count))

print("Checking Hand Landmark Regressor for BatchNorm2d layers...")
count = 0
for m in hand_regressor.modules():
        #print(m)
        if isinstance(m, torch.nn.BatchNorm2d):
            count += 1
            m.track_running_stats = False
            m.eval()
            print("BatchNorm2d layer {} is freezed".format(count))

print("Number of BatchNorm2d layers freezed: {}".format(count))
print("Done.")

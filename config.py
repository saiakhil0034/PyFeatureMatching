from utils.sift_features import SIFT
import cv2

DATA_FOLDER = "./data"    # You need to keep all data in this directory
MIN_MATCH_COUNT = 10      # for feature matching
data_collection = "viewpoints"  # name of data collection
data_object = "chatnoir"  # name of particular object whose images you are going to use
clr_o_grey = True  # Boolean to indicate whther we are processing grey or colour image
feature_point_detector = SIFT()  # Interesting point detector

# for feature matching filter
num_checks = 100
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=num_checks)
flann = cv2.FlannBasedMatcher(index_params, search_params)
feature_matcher = flann

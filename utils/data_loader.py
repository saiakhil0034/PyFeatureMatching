import cv2
import numpy as np


class DataLoader(object):
    """classs for all dataloader utilities"""

    def __init__(self):
        """constructor"""
        super(DataLoader, self).__init__()

    @staticmethod
    def get_data_path(root, collection, object_):
        """method to get the relevant data path"""
        return f"{root}/{collection}/{object_}/test"

    @staticmethod
    def load_img(path):
        """function to load an image"""
        return cv2.imread(path)

    @staticmethod
    def get_homography(path):
        """to get the relevant homography"""
        return np.loadtxt(homog_path, delimiter=',')


if __name__ == "__main__":
    homog_path = "./../data/viewpoints/chatnoir/test/homography/H1to2p"
    print(np.loadtxt(homog_path, delimiter=','))

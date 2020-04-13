import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.proc_img import ProcImg
from utils.data_loader import DataLoader

NUM_POINTS = 500


class SIFT(object):
    """class for SIFT"""

    def __init__(self, num_points=NUM_POINTS):
        """contructor"""
        super(SIFT, self).__init__()
        self.algo = cv2.xfeatures2d.SIFT_create(num_points)

    def get_key_points_n_discriptors(self, gray):
        """method for extracting SIFT features and their descriptors"""
        return self.algo.detectAndCompute(gray, None)

    def visualise(self, gray):
        """to visualise sift algorithm ouput"""
        kp, des = self.algo.detectAndCompute(gray, None)
        imgf = cv2.drawKeypoints(gray, kp, np.array([[]]))

        plt.figure(figsize=(12, 8))
        plt.imshow(imgf)
        plt.show()


if __name__ == "__main__":
    sift = SIFT(num_points=500)

    data_folder_path = "./../data/viewpoints/chatnoir/test"
    file_path = f"{data_folder_path}/image_color/img1.png"

    img = DataLoader.load_img(file_path)
    gray = ProcImg.get_gray_img(img)
    sift.visualise(gray)

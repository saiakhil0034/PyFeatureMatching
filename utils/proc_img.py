import cv2


class ProcImg(object):
    """class for processing images"""

    def __init__(self):
        """constructor"""
        super(proc_img, self).__init__()

    @staticmethod
    def get_gray_img(img):
        """ to convert image to gray scale image"""
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

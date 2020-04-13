import cv2


class FeatureMatching(object):
    """object/class for FeatureMatching"""

    def __init__(self):
        """constructor"""
        super(FeatureMatching, self).__init__()

    def find_homography(src_pts, dst_pts, algo=cv2.RANSAC):
        """for finding homography"""
        return cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

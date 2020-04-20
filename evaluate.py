import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from utils.data_loader import DataLoader
from utils.proc_img import ProcImg

from config import *


if __name__ == "__main__":

    data_folder_path = DataLoader.get_data_path(
        DATA_FOLDER, data_collection, data_object)
    img_dir = f'{data_folder_path}/image_{"color" if clr_o_grey else "gray"}'
    homg_dir = f'{data_folder_path}/homography'
    print(f"available images : {os.listdir(img_dir)}\n")

    img = DataLoader.load_img(f'{img_dir}/img1.png')
    gray = ProcImg.get_gray_img(img)
    kp, des = feature_point_detector.get_key_points_n_discriptors(gray)
    feature_point_detector.visualise(gray)
    imgf = cv2.drawKeypoints(gray, kp, img)

    for file in sorted(os.listdir(img_dir)):
        if file != "img1.png":
            img2_num = file[3]
            print(f"loading img : {file}")
            img2 = DataLoader.load_img(f'{img_dir}/{file}')
            gray2 = ProcImg.get_gray_img(img2)
            kp2, des2 = feature_point_detector.get_key_points_n_discriptors(
                gray2)

            img2f = cv2.drawKeypoints(gray2, kp2, img2)

            plt.figure(figsize=(14, 7))
            plt.subplot(121)
            plt.imshow(imgf)
            plt.subplot(122)
            plt.imshow(img2f)
            plt.show()

            matches = feature_matcher.knnMatch(
                des, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
            good = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good.append(m)

            if len(good) > MIN_MATCH_COUNT:
                src_pts = np.float32(
                    [kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32(
                    [kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                # estimating homography
                M, mask = cv2.findHomography(
                    src_pts, dst_pts, cv2.RANSAC, ransac_outlier_thresh)
                matchesMask = mask.ravel().tolist()
                print(f" estimated homography :\n{M}\n")
                real_homog = np.loadtxt(
                    f"{homg_dir}/H1to{img2_num}p", delimiter=',')
                print(f" real homography :\n{real_homog}\n")

                # calculating error based on frobenius norm
                error = ((M - real_homog)**2).sum()
                print(f"error in estimation : {error}")

                h, w = gray.shape
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                                  [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                img2 = cv2.polylines(
                    img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

            else:
                print(
                    f"Not enough matches are found - {len(good)}, {MIN_MATCH_COUNT}")
                matchesMask = None

            # visualising homography
            draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                               singlePointColor=None,
                               matchesMask=matchesMask,  # draw only inliers
                               flags=2)

            img3 = cv2.drawMatches(img, kp, img2, kp2,
                                   good, None, **draw_params)

            plt.imshow(img3, 'gray'), plt.show()

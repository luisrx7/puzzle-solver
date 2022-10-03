import numpy as np
import cv2 as cv
import argparse
import matplotlib.pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-q", "--queryimage", required = True, help = "Path to query image")
ap.add_argument("-t", "--targetimage", required = True, help = "Path to target image")
args = vars(ap.parse_args())

img1 = cv.imread(args["queryimage"], cv.IMREAD_GRAYSCALE)
img2 = cv.imread(args["targetimage"], cv.IMREAD_GRAYSCALE)

# cv.imshow("image1", image1)
# cv.imshow("image2", image2)
# cv.waitKey(0)

# Initiate SIFT detector
sift = cv.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# BFMatcher with default params
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
# cv.drawMatchesKnn expects list of lists as matches.
img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.imshow(img3),plt.show()

# # Initiate ORB detector
# orb = cv.ORB_create()
#
# # Find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(image1, None)
# kp2, des2 = orb.detectAndCompute(image2, None)
#
# # Create BFMatcher object
# bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
#
# # Match descriptors
# matches = bf.match(des1,des2)
#
# # Sort them in the order of their distance
# matches = sorted(matches, key = lambda x:x.distance)
#
# # Draw first 10 matches.
# image3 = cv.drawMatches(image1,kp1,image2,kp2,matches[:10],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# plt.imshow(image3)
# plt.show()

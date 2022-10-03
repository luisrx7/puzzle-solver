
'''

NOT TESTED YET

STILL IN ACTIVE DEVELOPMENT

'''

import os
import cv2
import sys
import math
import numpy as np
from sector import Sector

def assemble_puzzle (puzzleName):
    # Preprocess the global image
    sift = cv2.xfeatures2d.SIFT_create()
    img2 = cv2.imread('input/' + puzzleName + '/global.jpg')  # trainImage
    kp2, des2 = sift.detectAndCompute(img2, None)

    # Get puzzle info from JSON or other file
    (xdim, ydim) = (6, 4)
    pieceWidth = 2600  # necessary? in pixels or mm?
    pieceHeight = 2600

    # Zero out the board
    board = []
    for i in range(xdim):
        newCol = []
        for j in range(ydim):
            newCol.append(Sector())
        board.append[newCol]

    # Open video camera
    # cap = cv.VideoCapture(1); cap.open();

    # Prepare to loop
    done = False # poor form, reword
    count = 0

    while (done is not True):
        # img = cap.grab()  # From the webcam

        # Black out sectors that have already been pinned
        for x in range(xdim):
            for y in range(ydim):
                if (board[x][y].hasPiece):  # may need to switch x and y
                    canvas[pieceHeight*y:pieceHeight*(y+1), pieceWidth*x:pieceWidth*(x+1)] = np.zeros((pieceHeight, pieceWidth))

        # Prepare image and find contours (outlines of each puzzle piece)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        img = cv2.bilateralFilter(img, 9, 50, 50)
        ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        edgeDetected = cv2.Canny(thresh, 100, 200)  # Tune parameters?
        image, contours, hierarchy = cv2.findContours(edgeDetected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Draw the contours just as a reminder (or for the UI)
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)  # Go back to BGR space to draw contours in color
        blank = cv2.drawContours(img, contours, -1, (0,0,255), 10)

        # Draw white gridlines
        rows, cols = blank.shape
        for i in range(4):
            cv2.line(blank, (0,i*rows/ydim), (cols, i*rows/ydim), 255, 5, cv2.LINE_4)
        for j in range (6):
            cv2.line(blank, (j*cols/xdim, 0), (j*cols/xdim, rows), 255, 5, cv2.LINE_4)
        # TODO: Draw gridlines only over the target area

        cv2.imwrite('output/' + str(count) + '-contours.jpg', blank)  # Draw the pieces, contours, and gridlines

        # Get the first contour
        cnt = contours[0]
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  # Draw bounding rectangle for first contour

        # Crop the image
        # stg stg PIL # how to crop in opencv?

        # Discretize ... something?
        (ydim, xdim, colors) = img.shape
        (xdim, ydim) = (xdim/6, ydim/4)
        i = int(math.floor(x/xdim))
        j = int(math.floor(y/ydim))
        k = int(math.floor((x+w)/xdim))
        l = int(math.floor((y+h)/ydim))
        print i, j, k, l

        locationList = []
        for a in range(i,k+1):
            for b in range(j,l+1):
                print (a, b)
                locationList.append((a,b))
                board[a][b].pieces.append((x+0.5*w, y+0.5h))  # Early on: identify a piece by its centroid


        # Prepare to loop again
        count += 1
        done = True

    print '\a'


if __name__ == "__main__":
    puzzleName = sys.argv[1]
    assemble_puzzle(puzzleName)
    print '\a\a\a'

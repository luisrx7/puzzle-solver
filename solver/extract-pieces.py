import os
import cv2
import sys
import math
import numpy as np
from piece import Piece
from sector import Sector

PIECE_SIZE = 3000

if __name__ == '__main__':
    # Read in puzzle name and files
    puzzleName = sys.argv[1]
    inputFolder = 'input/' + puzzleName + '/all/'
    files = sorted(os.listdir(inputFolder))
    if '.DS_Store' in files:
        files.remove('.DS_Store')

    for f in files:
        img = cv2.imread(inputFolder + f)
        img = cv2.medianBlur(img, 9)
        img = cv2.bilateralFilter(img, 5, 50, 50)

        # Crop a little so more manageable
        xmax, ymax, colors = img.shape
        factor = 0.025
        img = img[int(factor*ymax):int((1-factor)*ymax), int(factor*xmax):int((1-factor)*xmax)]

        # Find the contours
        imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(imggray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Get only the good contours
        ct = []
        for c in contours:
            if (cv2.contourArea(c) < 0.75*xmax*ymax) and (cv2.contourArea(c) > 0.004*xmax*ymax):
                ct.append(c)
        print len(contours), len(ct)
        img3 = img.copy()
        img3 = cv2.drawContours(img, ct, -1, (0,255,0), 3)

        # Draw boxes around the pieces
        for c in ct:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            img3 = cv2.drawContours(img3,[box],0,(0,0,255),2)

        # Find boundaries and extract the piece
        count = 'a'
        for c in ct:
            # Get extrema of each contour
            leftmost = tuple(c[c[:,:,0].argmin()][0])
            rightmost = tuple(c[c[:,:,0].argmax()][0])
            topmost = tuple(c[c[:,:,1].argmin()][0])
            bottommost = tuple(c[c[:,:,1].argmax()][0])

            # Draw circles to visualize and make sure
            cv2.circle(img3, leftmost, 25, 255, -15)
            cv2.circle(img3, rightmost, 25, 255, -15)
            cv2.circle(img3, topmost, 25, 255, -15)
            cv2.circle(img3, bottommost, 25, 255, -15)

            # Extract piece from the all-pieces image
            img = cv2.imread(inputFolder + f)
            img = img[int(factor*ymax):int((1-factor)*ymax), int(factor*xmax):int((1-factor)*xmax)]
            paddingFactor = 0.075
            (top, left) = ( int(topmost[1]*(1-paddingFactor)), int(leftmost[0]*(1-paddingFactor)) )
            (bottom, right) = ( int(bottommost[1]*(1+paddingFactor)), int(rightmost[0]*(1+paddingFactor)) )
            img4 = img[top:bottom, left:right]

            # Uncrop the piece so it is square
            square = np.zeros((PIECE_SIZE, PIECE_SIZE, 3))
            m, n, c = img4.shape
            square[0:m, 0:n] = img4[:,:]

            # Write to file and continue
            cv2.imwrite('input/' + puzzleName + '/' + count + '.jpg', square)
            count = chr(ord(count) + 1)

        # Output global contours
        cv2.imwrite('output/' + puzzleName + '/contours-' + f, img3)

    print 'done'

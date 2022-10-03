import os
import cv2
import math
import numpy as np
from edge import Edge
import solve_helper

EDGE_TOLERANCE = 0.05
THRESHOLD_VALUE = 150

class Piece:
    count = 0
    alphabet = '`'

    def __init__ (self, puzzleName, imageFile, x=0, y=0, numMatches=0):
        #  Global variables
        self.puzzleName = puzzleName
        self.imageFile = imageFile
        self.outputFolder = "output/" + self.puzzleName + '/'

        # Variables that describe the piece itself
        Piece.count += 1
        Piece.alphabet = chr(ord(Piece.alphabet) + 1)
        self.pieceNumber = Piece.count
        self.pieceLetter = Piece.alphabet
        self.pieceName = os.path.splitext(imageFile)[0]   # TODO: Path now. Fix so it is just the letter?
        self.isPinned = False
        (self.x, self.y) = (x, y)
        self.numMatches = numMatches

        # Processing steps once it's created
        self.delete_table_background(True, False, False)
        # self.make_edge_images()  # If images are already cropped, comment this out to save time
        self.edges = self.categorize_piece(self.pieceName)
        # Edges are in order from left, bottom, right, top
        print self


    def __str__(self):
        line = self.pieceName + '  ' + str(self.pieceNumber) + '  (' + str(self.x) + ', ' + str(self.y) + ')  '
        for edge in self.edges:
            line += str(edge) + ' '
        line += ' matches:' + str(self.numMatches)
        return line


    def make_edge_images(self):
        corners = self.find_corners(self.imageFile)
        self.crop_edges(self.pieceName + "-mask.jpg", corners)


    ''' Finds the corners of a puzzle piece. '''
    def find_corners (self, filename):
        # Open file and prepare for analysis
        img1 = cv2.imread(filename)
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.medianBlur(img1, 5)
        img1 = cv2.bilateralFilter(img1, 9, 50, 50)

        # Threshold and find corners with Harris algorithm
        ret, thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imwrite(self.pieceName + '-thresh.jpg', thresh)

        # solve_helper.show(thresh)
        dst = cv2.cornerHarris(thresh, 5, 5, 0.03) #2, 3, 0.04) #5, 5, 0.01) # blocksize, ksize, k  # 5, 5, 0.1
        dst = cv2.dilate(dst, None); dst = cv2.dilate(dst, None);  # Why dilate? unclear TBH
        # If it is over a maximum value, then color it red (BGR) ... threshold value is unclear
        maxval = dst.max(); cornerThreshold = 0.01; maxCorner = maxval * cornerThreshold;

        img = cv2.imread(filename)
        img[dst > maxCorner] = [0,0,255] # Left is unclear and strange syntax.
        cv2.imwrite(self.pieceName + '-corners.jpg',img)

        # Get a list of all the corners
        cornerList=[]
        xmax, ymax = dst.shape
        for x in range(0, xmax, 5):
            for y in range (0, ymax, 5):
                if (dst[x,y] > maxCorner):
                    cornerList.append([y, x])

        # Find the closest corner in each semicardinal direction
        topLeft = self.find_closest_corner_to((0,0), cornerList, xmax, ymax)
        topRight = self.find_closest_corner_to((xmax, 0), cornerList, xmax, ymax)
        bottomLeft = self.find_closest_corner_to((0, ymax), cornerList, xmax, ymax)
        bottomRight = self.find_closest_corner_to((xmax, ymax), cornerList, xmax, ymax)

        return [topLeft, topRight, bottomLeft, bottomRight]


    ''' Find a specific corner of the puzzle piece '''
    def find_closest_corner_to (self, testCorner, cornerList, xmax, ymax):
        currentMinPoint = (0,0)
        currentMinDistance = self.distance((xmax,ymax), (0,0)) # max possible distance
        lowerLimit = self.distance((0.1*xmax,0.1*ymax), (0,0))

        # Don't pull out corners at the extremes (to account for the jumbling)
        (lowerXlimit, lowerYlimit) = (EDGE_TOLERANCE * xmax, EDGE_TOLERANCE * ymax)
        (maxXlimit, maxYlimit) = ((1-EDGE_TOLERANCE) * xmax, (1-EDGE_TOLERANCE) * ymax)

        for corner in cornerList:
            d = self.distance(corner, testCorner);
            if (d < currentMinDistance and d > lowerLimit and
            corner[0] > lowerXlimit and corner[1] > lowerYlimit and
            corner[0] < maxXlimit and corner[1] < maxYlimit):
                currentMinDistance = d
                currentMinPoint = corner
        return currentMinPoint


    ''' Euclidian distance between two (x, y) points '''
    def distance (self, point1, point2):
        xdist = (point2[0] - point1[0]) ** 2
        ydist = (point2[1] - point1[1]) ** 2
        return math.sqrt(xdist + ydist)


    ''' Crops the four edges (left, bottom, right, top) out of puzzle piece's image. '''
    def crop_edges (self, filename, corners):
        # Open the file to crop
        im = cv2.imread(filename)

        # Pull out variables to make math easier
        (x_max, y_max, colors) = im.shape
        (x1, y1) = (corners[0][0], corners[0][1])
        (x2, y2) = (corners[1][0], corners[1][1])
        (x3, y3) = (corners[2][0], corners[2][1])
        (x4, y4) = (corners[3][0], corners[3][1])

        interiorFactor = 0.5  # How much of the piece's interior to show (for female edges)
        left = (0, y1, int(x3 + interiorFactor*(x4 - x3)), y3)
        bottom = (x3, int(y3 - interiorFactor*(y3 - y1)), x4, y_max)
        right = (int(x2 - interiorFactor*(x2 - x1)), y2, x_max, y4)
        top = (x1, 0, x2, int(y2 + interiorFactor*(y4 - y2)))

        # Write it back to a file
        image = im[left[1]:left[3], left[0]:left[2]]
        cv2.imwrite(self.pieceName + "-1left.jpg", image)
        image = im[bottom[1]:bottom[3], bottom[0]:bottom[2]]
        cv2.imwrite(self.pieceName + "-2bottom.jpg", image)
        image = im[right[1]:right[3], right[0]:right[2]]
        cv2.imwrite(self.pieceName + "-3right.jpg", image)
        image = im[top[1]:top[3], top[0]:top[2]]
        cv2.imwrite(self.pieceName + "-4top.jpg", image)


    ''' Determines sex of all the edges of a piece. '''
    def categorize_piece (self, folder):
        left = Edge(folder + "-1left.jpg", -90, self)
        bottom = Edge(folder + "-2bottom.jpg", -180, self)
        right = Edge(folder + "-3right.jpg", -270, self)
        top = Edge(folder + "-4top.jpg", 0, self)
        return [left, bottom, right, top]


    ''' Draw the masked image and show other visual debug information'''
    def delete_table_background(self, mask=True, edges=True, contours=True):
        # Read in image and prepare for processing
        img = cv2.imread(self.imageFile)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.medianBlur(img, 5)
        img = cv2.bilateralFilter(img, 9, 50, 50)
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Threshold, detect edges, and find the contours
        if edges is True:
            edgeDetected = cv2.Canny(thresh, 100, 200) # Need to tune parameters
            cv2.imwrite(self.pieceName + '-edges.jpg', edgeDetected)
            image, contours, hierarchy = cv2.findContours(edgeDetected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if contours is True:
                # Find the edge of the piece (assume that it's the biggest contour)
                maxValue = 0
                biggestContour = None
                for c in contours:
                    c_val = cv2.arcLength(c, True)  # Area or perimeter? Or both?
                    # c_val = cv2.contourArea(c)
                    if (c_val > maxValue):
                        maxValue = c_val
                        biggestContour = c

                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Go back to BGR space to draw contours in color
                blank = cv2.drawContours(img, [biggestContour], 0, (0,0,255), 5)
                cv2.imwrite(self.pieceName + '-contours.jpg', blank)

        if mask is True:
            canvas = cv2.imread(self.imageFile)
            canvas[thresh == 0] = [0,0,0]
            cv2.imwrite(self.pieceName + '-mask.jpg', canvas)


    ''' Rotate the piece based on image moments (not content features) '''
    @staticmethod
    def rotate_piece (img1, puzzleName, letter):
        # Greyscale, blur, threshold, detect edges, and get contours
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img1 = cv2.medianBlur(img1, 5)
        img1 = cv2.bilateralFilter(img1, 9, 50, 50)
        blur = cv2.GaussianBlur(img1, (9,9), 0)
        return_value, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edgeDetected = cv2.Canny(thresh, 100, 200)  # Tune parameters?
        image, contours, hierarchy = cv2.findContours(edgeDetected, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Find the main contour (around the puzzle piece)
        (area, ct) = (0, None)
        for c in contours:
            if cv2.contourArea(c) > area:
                area = cv2.contourArea(c)  # Maybe longest perimeter, centroid nearest the center, or otherwise
                ct = c

        # Use the image moments of the main contour to determine how it is rotated
        M = cv2.moments(ct)
        (cx, cy) = ( int(M['m10']/M['m00']), int(M['m01']/M['m00']) )
        m_prime_20 = M['m20']/M['m00'] - cx*cx
        m_prime_02 = M['m02']/M['m00'] - cy*cy
        m_prime_11 = M['m11']/M['m00'] - cx*cy
        # Reference: https://en.wikipedia.org/wiki/Image_moment
        theta = math.degrees(0.5 * math.atan(2*m_prime_11 / (m_prime_20 - m_prime_02)))

        # Draw contour, centroid, and theta for debugging purposes
        img = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        b = cv2.drawContours(img, [ct], 0, (255,0,0), 5)
        cv2.circle(b, (cx, cy), 35, 255, -1)
        # TODO: Draw the theta as two lines
        outputFile = 'output/' + puzzleName + '/no-features/' + letter
        cv2.imwrite(outputFile + '-analysis.jpg', b)

        # Find eigen-values and eccentricity (optional)
        p1 = (m_prime_20 + m_prime_02)/2.0
        p2 = math.sqrt(4*m_prime_11*m_prime_11 + (m_prime_20 - m_prime_02)**2) * 0.5
        (lambda1, lambda2) = (p1 + p2 , p1 - p2)
        ratio = lambda1/lambda2
        if ratio > 1:
            ratio = 1/ratio
        eccentricity = math.sqrt(1 - ratio)  # e=0 for a circle, 0<e<1 for an ellipse, e=1 parabola
        print "eccentricity", eccentricity

        # Rotate image back and output it
        rows, cols = img1.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2), theta, 1)
        image3 = cv2.warpAffine(img1, M, (cols, rows))
        image3 = cv2.cvtColor(image3, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(outputFile + '-rotated.jpg', image3)
        return image3

import os
import cv2
import sys
import math
import numpy as np
from piece import Piece
from sector import Sector

MIN_MATCH_COUNT = 4
USE_MASKED_IMAGE = False
DORY_DIMENSIONS = (6, 4)
MINION_DIMENSIONS = (3, 4)

def draw_puzzle(puzzleName, board, height, width):
    # Zero out the canvas and prepare for drawing
    canvas = np.zeros((len(board[0]) * width, len(board) * height, 3))
    count = 0
    (w, h) = (len(board), len(board[0]))

    # Go through matrix one-by-one. If there is a piece there, then draw it.
    for i in range(w):
        for j in range(h):
            if board[i][j] is not None:
                piece_to_draw = board[i][j].pieces[0]  # First piece in the list will have most matches
                readFile = piece_to_draw.pieceName + ("-mask.jpg" if USE_MASKED_IMAGE is True else ".jpg")
                img = cv2.imread(readFile, cv2.IMREAD_REDUCED_COLOR_4)
                resized = cv2.resize(img, (width, height), interpolation = cv2.INTER_CUBIC)

                if USE_MASKED_IMAGE is True:  # Draw the images overlapped
                    x_start = int(width * 0.5 * i)
                    x_end = x_start + width
                    y_start = int(height * 0.5 * j)
                    y_end = y_start + height
                    canvas[y_start:y_end, x_start:x_end] += resized[:,:]
                    # TODO: Crop canvas to remove extra space
                else:  # Draw the images side-by-side
                    canvas[height*j:height*(j+1), width*i:width*(i+1)] = resized[:,:]

                # Draw a circle to indicate there is more than 1 piece in that sector
                if len(board[i][j].pieces) > 1:
                    x = int(width * (i + 0.5))
                    y = int(height * (j + 0.5))
                    cv2.circle(canvas, (x,y), 75, 255, -15)

                count += len(board[i][j].pieces)  # Should be just one, but sometimes more

    # Draw background in white
    # if count > 22:
    #     h, w, c = canvas.shape
    #     for i in range(h):
    #         for j in range(w):
    #             if (canvas[i][j][0] == 0 and canvas[i][j][1] == 0 and canvas[i][j][2] == 0):
    #                 canvas[i][j] = [255, 255, 255]

    resized = cv2.resize(canvas, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imwrite('output/'+ puzzleName + "/canvas/" + str(count) + '.jpg', resized)


def integrate_shape_solver(puzzleName, board, height, width):
    (w, h) = (len(board), len(board[0]))

    print '\nOTHER PIECES\n'
    # Get pieces that were placed in a sector, but the sector already has a piece with more matches
    extraPieces = []
    for i in range(w):
        for j in range(h):
            if (board[i][j] is not None) and (len(board[i][j].pieces) > 1):
                for p in range(1, len(board[i][j].pieces)):
                    print board[i][j].pieces[p]
                    extraPieces.append(board[i][j].pieces[p])
    print "Extra pieces", len(extraPieces)

    emptySectors = []
    for i in range(w):
        for j in range(h):
            if board[i][j] is None:
                shouldBe = {}
                shouldBe['left'] = lookinDirection(board, i, j, 'left')
                shouldBe['bottom'] = lookinDirection(board, i, j, 'bottom')
                shouldBe['right'] = lookinDirection(board, i, j, 'right')
                shouldBe['top'] = lookinDirection(board, i, j, 'top')
                shouldBe['coordinates'] = (i, j)
                dof = 0  # Degrees of freedom, or how constrained the empty sector is
                for key in shouldBe:
                    if shouldBe[key] is 'x':
                        dof += 1
                shouldBe['dof'] = dof
                emptySectors.append(shouldBe)
                print shouldBe
    emptySectors = sorted(emptySectors, key=lambda sector: sector['dof'])
    # Sort so the pieces with the least degrees of freedom (most constrained) are first

    # Try pinning extra pieces to sectors with 0 and 1 dof's
    for s in emptySectors:
        print 'coordinates', s['coordinates']
        if s['dof'] < 2:  # Don't worry about others for now (maybe recursive or a diff while loop)
            for p in extraPieces:
                sexmatches = 0
                sexmatches += 1 if p.edges[0].sex is s['left'] else 0
                sexmatches += 1 if p.edges[1].sex is s['bottom'] else 0
                sexmatches += 1 if p.edges[2].sex is s['right'] else 0
                sexmatches += 1 if p.edges[3].sex is s['top'] else 0
                print p.pieceLetter, sexmatches
                # Try all four rotations of a piece at a time
                # If it's a 4, then definitely a yes (for now, no shape-fitting)
                # Would 3 be a success now?

    # Randomly assign all remaining pieces (just in case feature-determ or shape-determ was poor)
    for i in range(w):
        for j in range(h):
            if (board[i][j] is None) and (len(extraPieces) > 0):
                board[i][j] = Sector(True, extraPieces.pop())

    draw_puzzle(puzzleName, board, height, width)


def lookinDirection (board, i, j, direction):
    # If you are already on the puzzle perimeter, then you're looking for a neutral/flat edge
    if (direction is 'left' and i is 0) or (direction is 'bottom' and i is len(board)) or \
       (direction is 'right' and j is len(board[0])) or (direction is 'top' and j is 0):
        return 'n'

    indices = {'left': (-1, 0), 'bottom': (0, 1), 'right': (1, 0), 'top':(0, -1)}
    (x, y) = indices[direction]
    if board[i+x][j+y] is None:
        return 'x'  # This sector doesn't have a piece too, so it is a 'degree of freedom'

    # Get piece and edge to compare and return it's opposite
    testPiece = board[i+x][j+y].pieces[0]
    otherIndex = {'left': 2, 'bottom': 3, 'right': 0, 'top': 1}
    a = otherIndex[direction]
    if testPiece.edges[a].sex is 'm':
        return 'f'
    elif testPiece.edges[a].sex is 'f':
        return 'm'
    else:
        return 'x'


if __name__ == "__main__":
    # Read in puzzle name and files
    puzzleName = sys.argv[1]
    inputFolder = "input/" + puzzleName
    files = sorted(os.listdir(inputFolder))
    removeFiles = ['.DS_Store', 'global', 'all']
    for f in removeFiles:
        if f in files:
            files.remove(f)

    # Find features on the large, global image
    sift = cv2.xfeatures2d.SIFT_create()
    img1 = cv2.imread('input/' + puzzleName + '/global/global.jpg')  # Global image
    kp2, des2 = sift.detectAndCompute(img1, None)
    img2 = cv2.imread('input/' + puzzleName + '/global/global.jpg', cv2.IMREAD_GRAYSCALE)  # For centroid check

    # Zero out the board
    (xdim, ydim) = DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
    board = [[None for j in range(ydim)] for i in range(xdim)]
    success_count = 0

    for afile in files:
        # Read in query image, find features, and find matches
        img3 = cv2.imread('input/' + puzzleName + '/' + afile, cv2.IMREAD_REDUCED_COLOR_2)  # Query image
        kp1, des1 = sift.detectAndCompute(img3, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to only pick out the 'good' feature matches
        good = []
        points = []
        for m,n in matches:
            if m.distance < 0.70 * n.distance:  # May need to modify to get most # of pins
                good.append([m])
                points.append([int(kp2[m.trainIdx].pt[0]), int(kp2[m.trainIdx].pt[1])])
        points = np.float32(points)

        # Use k-means clustering to find where the most matches are
        k = 4
        if k > len(points):
            k = len(points)  # If less matches than 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, label, center = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Separate the data into clusters, find cluster with most elements, and get its centroid
        series = []
        for i in range(k):
            series.append(points[label.ravel()==i])
        mostElements = series[0]
        for s in series:
            if len(s) > len(mostElements):
                mostElements = s
        (x, y) = (int(np.mean(mostElements[:,0])), int(np.mean(mostElements[:,1])))

        # Filter to keep just the good feature matches in the best cluster
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                if (int(kp2[m.trainIdx].pt[0]) in mostElements[:,0]):
                    good.append(m)

        # Rotate the piece back to the correct orientation
        img4 = img1.copy()  # Copy of global image for feature-matching
        img5 = None  # Correctly rotated version of the piece
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Estimate an affine transform
            Mprime = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            degrees = math.degrees(math.atan2(Mprime[0][1][0], Mprime[0][0][0])) * (-1.0)

            # Rotate the piece back
            rows, cols, colors = img3.shape
            M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
            img5 = img3.copy()
            img5 = cv2.warpAffine(img3, M, (cols, rows))
        else:
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            img5 = Piece.rotate_piece(img3, puzzleName, afile)
        draw_params = dict(matchColor = (0, 255, 0),
           singlePointColor = None,
           matchesMask = matchesMask,  # Only draw the inliers
           flags = 2)
        img4 = cv2.drawMatches(img3, kp1, img4, kp2, good, img4, **draw_params)

        # Draw the rotated version below original. Write to directory for further analysis.
        readFile = "output/" + puzzleName + "/rotated/" + afile
        img4[rows:2*rows, 0:cols] = img5
        cv2.imwrite(readFile, img5)

        # Find the centroid of the cluster
        (ydim, xdim, colors) = img1.shape
        (xx, yy) = DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
        (xdim, ydim) = (xdim/xx, ydim/yy)
        (xcoord, ycoord) = (int(math.floor(x/xdim)), int(math.floor(y/ydim)))
        x = int(math.floor(x/xdim)*xdim + xdim/2)  # Discretize the centroid
        y = int(math.floor(y/ydim)*ydim + ydim/2)
        cv2.circle(img2, (x,y), 50, 255, -15)  # Draw centroid on b/w global image
        rows, cols, colors = img3.shape
        x += cols  # Offset centroid to account for the side-by-side image
        cv2.circle(img4, (x,y), 50, 255, -15)  # Draw centroid in blue for this specific piece output

        # Communicate a success (lots of matches), tally, and write to file
        if (len(mostElements) > MIN_MATCH_COUNT):
            success_count += 1
        resized = cv2.resize(img4, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite("output/" + puzzleName + "/features/" + afile, resized)

        # Create a new piece, put in the right spot, and draw on canvas
        newPiece = Piece(puzzleName, readFile, xcoord, ycoord, len(good))
        if board[xcoord][ycoord] is None:
            board[xcoord][ycoord] = Sector(True, newPiece)
        else:
            board[xcoord][ycoord].pieces.append(newPiece)
            board[xcoord][ycoord].pieces = sorted(board[xcoord][ycoord].pieces,
                reverse=True, key=lambda piece: piece.numMatches)
        height, width = img3.shape[:2]
        draw_puzzle(puzzleName, board, height, width)
        print ' '

    # Draw white gridlines on the global centroid image
    rows, cols = img2.shape
    (xdim, ydim) = DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
    for i in range(ydim):
        cv2.line(img2, (0,i*rows/ydim), (cols, i*rows/ydim), 255, 5, cv2.LINE_4)
    for j in range (xdim):
        cv2.line(img2, (j*cols/xdim, 0), (j*cols/xdim, rows), 255, 5, cv2.LINE_4)
    cv2.imwrite("output/" + puzzleName + "/all-centroids.jpg", img2)

    # Wrap-up
    print "successfully found", success_count, '\n'
    integrate_shape_solver(puzzleName, board, height, width)

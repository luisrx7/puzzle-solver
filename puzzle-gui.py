import os
import cv2
import sys
import math
import numpy as np
from piece import Piece
from sector import Sector
from PIL import Image, ImageTk
import tkinter as tk

PADDING_X = 15
PADDING_Y = 10

class puzzleGUI:

    def __init__(self, master):
        self.des2 = None  # Matches on global image
        self.kp2 = None
        self.success_count = 0
        self.pieces_count = 0
        self.board = None
        self.master = master
        self.setupGui(master)


    def setupGui (self, master):
        # Create GUI elements
        self.recent_label = tk.Label(master, text='Most Recent Piece', padx=PADDING_X, pady=PADDING_Y)
        self.analysis_label = tk.Label(master, text='Target & Analysis')
        self.recent_image = tk.Label(master, text=' ')
        self.analysis_image = tk.Label(master, text=' ')
        self.getfile_button = tk.Button(master, text='Import piece from file', command=master.quit)
        self.getglobal_button = tk.Button(master, text='Import poster image from file', command=master.quit)

        self.camera_label = tk.Label(master, text='Camera Feed')
        self.camera_image = tk.Label(master, text='Live Camera Feed')
        self.capture_button = tk.Button(master, text='Capture', command=master.quit)
        self.quit_button = tk.Button(master, text='Quit', command=master.quit)

        # Place GUI elements
        self.recent_label.grid(row=0, column=0)
        self.analysis_label.grid(row=0, column=1)
        self.recent_image.grid(row=1, column=0)
        self.analysis_image.grid(row=1, column=1)
        self.getfile_button.grid(row=2, column=0)
        self.getglobal_button.grid(row=2, column=1)

        self.camera_label.grid()
        self.camera_image.grid()
        self.capture_button.grid()
        self.quit_button.grid()


    def start(self):
        # Get a global image
        name = tkFileDialog.askopenfilename(initialdir = '~/Dropbox/Puzzle/solver/input',
        title = 'Select Global Puzzle Image',
        filetypes = (('jpeg files','*.jpg'), ('all files','*.*')))
        subpaths = name.split('/')
        self.puzzle_name_label.configure(text=subpaths[-1])
        puzzleName = subpaths[-3] + '/' + subpaths [-2] + '/' + subpaths[-1]

        # Find SIFT features on the global image
        sift = cv2.xfeatures2d.SIFT_create()
        trainImage = cv2.imread(puzzleName)  # Global image
        self.trainImage = trainImage
        self.kp2, self.des2 = sift.detectAndCompute(trainImage, None)

        # Reset the board and puzzle data to 0
        (xdim, ydim) = DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
        dims = '(' + str(xdim) + ' x ' + str(ydim) + ')'
        self.puzzle_dimensions_label.configure(text=dims)
        # TODO: make getting the dimensions more interactive
        self.board = [[None for j in range(ydim)] for i in range(xdim)]
        self.success_count = 0
        self.pieces_count = 0

        # Enable buttons
        self.delete_button.configure(state=DISABLED)
        self.capture_button.configure(state=NORMAL)
        self.file_button.configure(state=NORMAL)
        self.folder_button.configure(state=NORMAL)


    def getFile(self):
        name = tkFileDialog.askopenfilename(initialdir = '~/Dropbox/Puzzle/solver/input',
        title = 'Select Individual Puzzle Piece',
        filetypes = (('jpeg files','*.jpg'),('all files','*.*')))
        subpaths = name.split('/')
        self.piecename_label.configure(text=subpaths[-1])
        piecesPath = subpaths[-4] + '/' + subpaths[-3] + '/' + subpaths [-2] + '/'
        pieceName = subpaths[-1]
        puzzleName = subpaths[-2]
        self.analyze_piece(pieceName, piecesPath, puzzleName)



    def analyze_piece(self, pieceName, piecesPath, puzzleName):
        print (piecesPath + pieceName)

        # Read in query image, find features, and find matches
        queryImage = cv2.imread(piecesPath + pieceName, cv2.IMREAD_REDUCED_COLOR_2)
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(queryImage, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, self.des2, k=2)

        # Apply ratio test to only pick out the 'good' feature matches
        good = []; points = [];
        for m,n in matches:
            if m.distance < 0.70 * n.distance:  # May need to modify to get most # of pins
                good.append([m])
                points.append([int(self.kp2[m.trainIdx].pt[0]), int(self.kp2[m.trainIdx].pt[1])])
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
                if (int(self.kp2[m.trainIdx].pt[0]) in mostElements[:,0]):
                    good.append(m)
        self.matches_label.configure(text="Matches: "+str(len(good)))

        # Rotate the piece back to the correct orientation
        img4 = self.trainImage  # Copy of global image for feature-matching
        rows, cols, colors = queryImage.shape
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Estimate an affine transform
            Mprime = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            degrees = math.degrees(math.atan2(Mprime[0][1][0], Mprime[0][0][0])) * (-1.0)
            self.rotation_label.configure(text="Rotation: " + str(round(degrees)))

            # Rotate the piece back
            M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
            img5 = queryImage.copy()
            img5 = cv2.warpAffine(queryImage, M, (cols, rows))
        else:
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
            img5 = Piece.rotate_piece(queryImage, puzzleName, pieceName)
        draw_params = dict(matchColor = (0, 255, 0),
           singlePointColor = None,
           matchesMask = matchesMask,  # Only draw the inliers
           flags = 2)
        img4 = cv2.drawMatches(queryImage, kp1, img4, self.kp2, good, img4, **draw_params)

        # Draw the rotated version below original. Write to directory for further analysis.
        readFile = "output/" + puzzleName + "/rotated/" + pieceName
        img4[rows:2*rows, 0:cols] = img5
        cv2.imwrite(readFile, img5)

        # Find the centroid of the cluster
        (ydim, xdim, colors) = self.trainImage.shape
        (xx, yy) = DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
        (xdim, ydim) = (xdim/xx, ydim/yy)
        (xcoord, ycoord) = (int(math.floor(x/xdim)), int(math.floor(y/ydim)))
        self.location_label.configure(text="Location: ("+str(xcoord+1)+", "+str(ycoord+1)+")")  # Start from 0 is unnatural
        x = int(math.floor(x/xdim)*xdim + xdim/2)  # Discretize the centroid
        y = int(math.floor(y/ydim)*ydim + ydim/2)
        rows, cols, colors = queryImage.shape
        x += cols  # Offset centroid to account for the side-by-side image
        cv2.circle(img4, (x,y), 50, 255, -15)  # Draw centroid in blue for this specific piece output

        # If lots of matches, tally and communicate a success
        if (len(mostElements) > MIN_MATCH_COUNT):
            self.success_count += 1
            self.success_count_label.configure(text='Successes: '+str(self.success_count))

        # Resize, write to file, and show on-screen.
        resized = cv2.resize(img4, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        resized = cv2.resize(img4, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite("output/" + puzzleName + "/analysis/" + pieceName, resized)
        image = Image.open("output/" + puzzleName + "/analysis/" + pieceName)
        photo = ImageTk.PhotoImage(image)
        self.analysis_image.configure(image=photo)
        self.analysis_image.photo = photo

        # Create a new piece, put in the right spot, and draw on canvas
        newPiece = Piece(puzzleName, readFile, xcoord, ycoord, len(good))
        if self.board[xcoord][ycoord] is None:
            self.board[xcoord][ycoord] = Sector(True, newPiece)
        else:
            self.board[xcoord][ycoord].pieces.append(newPiece)
            self.board[xcoord][ycoord].pieces = sorted(self.board[xcoord][ycoord].pieces,
                reverse=True, key=lambda piece: piece.numMatches)
        height, width = queryImage.shape[:2]
        self.draw_puzzle(puzzleName, self.board, height, width, pieceName)

        self.pieces_count += 1
        self.pieces_count_label.configure(text='Pieces: '+str(self.pieces_count))


    def delete_piece(self):
        pass

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Python Computer Vision Puzzle Solver')
    app = puzzleGUI(root)
    root.mainloop()

import os
import cv2
import sys
import math
import numpy as np
from Tkinter import *
from piece import Piece
from sector import Sector
from PIL import Image, ImageTk
import Tkinter, Tkconstants, tkFileDialog
import thread

PADDING_X = 15
PADDING_Y = 10
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 335
MIN_MATCH_COUNT = 4
USE_MASKED_IMAGE = False
DORY_DIMENSIONS = (6, 4)
MINION_DIMENSIONS = (3, 4)

class puzzleGUI:

    def __init__(self, master):
        self.des2 = None  # Matches on global image
        self.kp2 = None
        self.success_count = 0
        self.pieces_count = 0
        self.board = None
        self.master = master
        self.setupGui(master)


    def setupGui(self, master):
        # Create first half of GUI elements
        self.canvas_label = Label(master, text='Canvas', padx=PADDING_X, pady=PADDING_Y)
        self.canvas_image = Label(master, text=' ')
        self.global_label = Label(master, text='Global')
        self.puzzle_name_label = Label(master, text=' ')
        self.puzzle_dimensions_label = Label(master, text=' ')
        self.success_count_label = Label(master, text='Successes: '+str(self.success_count))
        self.pieces_count_label = Label(master, text='Pieces: '+str(self.pieces_count))
        self.start_button = Button(master, text='START', command=self.start)
        self.delete_button = Button(master, text='DELETE', command=self.delete_piece, state=DISABLED)
        self.quit_button = Button(master, text='QUIT', command=master.quit)

        # Place the first half of the GUI
        self.canvas_label.grid(sticky=W)
        self.canvas_image.grid(rowspan=7)
        self.global_label.grid(row=0, column=1, sticky=W)
        self.puzzle_name_label.grid(row=1, column=1)
        self.puzzle_dimensions_label.grid(row=2, column=1)
        self.success_count_label.grid(row=3, column=1)
        self.pieces_count_label.grid(row=4, column=1)
        self.start_button.grid(row=5, column=1)
        self.delete_button.grid(row=6, column=1)
        self.quit_button.grid(row=7, column=1)

        # Create second half of the GUI elements
        self.analysis_label = Label(master, text='Analysis', padx=PADDING_X, pady=PADDING_Y)
        self.analysis_image = Label(master, text=' ')
        self.piece_label = Label(master, text='Piece')
        self.piecename_label = Label(master, text=' ')
        self.matches_label = Label(master, text=' ')
        self.location_label = Label(master, text=' ')
        self.rotation_label = Label(master, text=' ')
        self.capture_button = Button(master, text='CAPTURE', command=master.quit, state=DISABLED)
        self.file_button = Button(master, text='IMPORT FILE', command=self.getFile, state=DISABLED)
        self.folder_button = Button(master, text='IMPORT FOLDER', command=self.getFolder, state=DISABLED)

        # Place the second half of the GUI
        self.analysis_label.grid(row=8, sticky=W)
        self.analysis_image.grid(row=9, rowspan=7)
        self.piece_label.grid(row=8, column=1, sticky=W)
        self.piecename_label.grid(row=9, column=1)
        self.matches_label.grid(row=10, column=1)
        self.location_label.grid(row=11, column=1)
        self.rotation_label.grid(row=12, column=1)
        self.capture_button.grid(row=13, column=1)
        self.file_button.grid(row=14, column=1)
        self.folder_button.grid(row=15, column=1)


    def clearGUI(self):
        self.puzzle_name.set('None')
        self.puzzle_dimensions.set('0 x 0')
        self.success_count = 0
        self.success_count_label.configure(text='Successes: '+str(self.success_count))
        self.pieces_count.set('Pieces: 0')

        self.piece_name.set('None')
        self.num_matches.set('Matches: 0')
        self.location.set('Location: ( , )')
        self.rotation.set('Rotation: 0 degrees')
        self.time.set('Time: 0 ms')
        pass


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


    def getFolder(self):
        name = tkFileDialog.askdirectory(initialdir= '~/Dropbox/Puzzle/solver/input', title='Select Folder')
        subpaths = name.split('/')
        self.piecename_label.configure(text=subpaths[-1])
        piecesPath = subpaths[-3] + '/' + subpaths[-2] + '/' + subpaths[-1] + '/'
        puzzleName = subpaths[-1]
        files = sorted(os.listdir(piecesPath))
        if '.DS_Store' in files:
            files.remove('.DS_Store')
        for f in files:
           # thread.start_new_thread(self.analyze_piece, (f, piecesPath, puzzleName))
           self.analyze_piece(f, piecesPath, puzzleName)


    def analyze_piece(self, pieceName, piecesPath, puzzleName):
        print piecesPath + pieceName

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


    def draw_puzzle(self, puzzleName, board, height, width, pieceName):
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

        resized = cv2.resize(canvas, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        cv2.imwrite('output/'+ puzzleName + "/canvas/" + str(count) + '.jpg', resized)

        resized = cv2.resize(canvas, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imwrite("output/" + puzzleName + "/canvas/" + 'mini-' + pieceName, resized)
        image = Image.open("output/" + puzzleName + "/canvas/" + 'mini-' + pieceName)
        photo = ImageTk.PhotoImage(image)
        self.canvas_image.configure(image=photo)
        self.canvas_image.photo = photo



if __name__ == '__main__':
    root = Tk()
    root.title('Python Computer Vision & Robot Puzzle Solver')
    app = puzzleGUI(root)
    root.mainloop()

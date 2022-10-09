import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import math
import myvideocapture as mvc

CAMERA_WIDTH = 563
CAMERA_HEIGHT = 450
MIN_MATCH_COUNT = 4
ICECREAMTRUCK_DIMENSIONS = (20,15)
CROP_FACTOR = 0.2

class puzzleGUI:

    def __init__(self, master, video_source=0):
        self.master = master
        self.query_image = None
        self.train_image = None
        self.kp2 = None
        self.des2 = None

        # Open video source (by default this will try to open the computer webcam)
        self.video_source = video_source
        self.cap=cv2.VideoCapture(0)
        self.vid = mvc.MyVideoCapture(self.video_source)

        self.setup_gui(master)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.master.mainloop()

    def setup_gui (self, master):
        # Create GUI elements
        self.recent_label = tk.Label(master, text='Most Recent Piece')
        self.analysis_label = tk.Label(master, text='Target Image')
        self.recent_image = tk.Label(master, text=' ')
        self.analysis_image = tk.Label(master, text=' ')
        self.getfile_button = tk.Button(master, text='Import', command=self.get_piece_from_file)
        self.getglobal_button = tk.Button(master, text='Import', command=self.get_poster_image)

        self.camera_label = tk.Label(master, text='Camera Feed')
        self.camera_image = tk.Label(master, text='Live Camera Feed')
        self.capture_button = tk.Button(master, text='Capture', command=self.snapshot)
        self.matches_label = tk.Label(master, text=' ')
        self.rotation_label = tk.Label(master, text=' ')
        self.location_label = tk.Label(master, text=' ')
        self.quit_button = tk.Button(master, text='Quit', command=master.quit)

        # Place GUI elements
        self.recent_label.grid(row=0, column=0)
        self.analysis_label.grid(row=0, column=1)
        self.recent_image.grid(row=1, column=0)
        self.analysis_image.grid(row=1, column=1)
        self.getfile_button.grid(row=2, column=0)
        self.getglobal_button.grid(row=2, column=1)

        self.camera_label.grid()
        self.camera_image.grid(rowspan=3)
        self.matches_label.grid(row=4, column=1)
        self.rotation_label.grid(row=5, column=1)
        self.location_label.grid(row=6, column=1)
        self.capture_button.grid(row=7, column=0)
        self.quit_button.grid(row=7, column=1)

    def crop_image (self, image):
        factor = CROP_FACTOR
        (width, height, color) = image.shape
        return image[:,int(factor*height):int(height*(1-factor))]

    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = self.crop_image(frame)

        if ret:
            photo = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((int(CAMERA_WIDTH*(1-2*CROP_FACTOR)), CAMERA_HEIGHT)))
            self.recent_image.configure(image=photo)
            self.recent_image.photo = photo
            self.query_image = frame
            self.analyze_piece()

    def get_poster_image(self):
        pil_photo, cv_photo = self.get_image_from_file()
        self.analysis_image.configure(image=pil_photo)
        self.analysis_image.photo = pil_photo
        self.train_image = cv_photo

        sift = cv2.SIFT_create()
        kp2, des2 = sift.detectAndCompute(self.train_image, None)
        # Get features of target image and  analyze them at load rather than with every new piece

    def get_piece_from_file(self):
        pil_photo, cv_photo = self.get_image_from_file()
        self.recent_image.configure(image=pil_photo)
        self.recent_image.photo = pil_photo
        self.query_image = cv_photo
        self.analyze_piece()

    def get_image_from_file(self):
        name = tk.filedialog.askopenfilename(initialdir = '../puzzle-solver/input',
        filetypes = (('jpeg files','*.jpg'),('all files','*.*')))

        load = Image.open(name)
        resized = load.resize((CAMERA_WIDTH, CAMERA_HEIGHT))
        pil_photo = ImageTk.PhotoImage(resized)

        cv_photo = cv2.imread(name)

        return pil_photo, cv_photo

    def analyze_piece (self):
        # Read in query image, find features, and then find matches
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.query_image, None)
        kp2, des2 = sift.detectAndCompute(self.train_image, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)

        # Apply ratio test to only pick out the 'good' feature matches
        good = []; points = [];
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
        if len(points) > 1:
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
        self.matches_label.configure(text="Matches: "+str(len(good)))

        # Rotate the piece back to the correct orientation
        img4 = self.train_image.copy()  # Copy of global image for feature-matching
        try: rows, cols, colors = self.query_image.shape
        except: rows, cols = self.query_image.shape # File input has 2 values, webcam has 3
        if len(good) >= MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            # Estimate an affine transform
            Mprime = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            degrees = math.degrees(math.atan2(Mprime[0][1][0], Mprime[0][0][0])) * (-1.0)
            text = str(round(abs(degrees)))
            if degrees > 0:
                text += " counterclockwise"
            else:
                text += " clockwise"
            self.rotation_label.configure(text="Rotation: " + text)

            # Rotate the piece back
            M = cv2.getRotationMatrix2D((cols/2,rows/2), degrees, 1)
            img5 = self.query_image.copy()
            img5 = cv2.warpAffine(self.query_image, M, (cols, rows))
        else:
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None

        draw_params = dict(matchColor = (0, 255, 0),
           singlePointColor = None,
           matchesMask = matchesMask,  # Only draw the inliers
           flags = 2)

        # Find the centroid of the cluster
        (ydim, xdim, colors) = self.train_image.shape
        (xx, yy) = ICECREAMTRUCK_DIMENSIONS # DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
        (xdim, ydim) = (xdim/xx, ydim/yy)
        (xcoord, ycoord) = (int(math.floor(x/xdim)), int(math.floor(y/ydim)))
        self.location_label.configure(text="Location: ("+str(xcoord+1)+", "+str(ycoord+1)+")")  # Start from 0 is unnatural
        x = int(math.floor(x/xdim)*xdim + xdim/2)  # Discretize the centroid
        y = int(math.floor(y/ydim)*ydim + ydim/2)

        try: rows, cols, colors = self.query_image.shape
        except: rows, cols = self.query_image.shape # File input has 2 values, webcam has 3

        cv2.circle(img4, (x,y), 150, 255, -15)  # Draw centroid in blue for this specific piece output

        # Resize and show on screen
        try: (x, y, c) = img4.shape
        except: (x, y) = img4.shape
        (newx, newy) = (int(math.floor(y*CAMERA_HEIGHT*1.0/x)), CAMERA_HEIGHT)
        resized = cv2.resize(img4, (newx, newy))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = ImageTk.PhotoImage(Image.fromarray(img))
        self.analysis_image.configure(image=im_pil)
        self.analysis_image.photo = im_pil

    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = self.crop_image(frame)
            photo = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((int(CAMERA_WIDTH*0.6), CAMERA_HEIGHT)))
            self.camera_image.configure(image=photo)
            self.camera_image.photo = photo
        self.master.after(self.delay, self.update)

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Python Computer Vision Puzzle Solver')
    app = puzzleGUI(root)

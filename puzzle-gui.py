import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog
import math
import myvideocapture as mvc
import time

CAMERA_WIDTH = 813
CAMERA_HEIGHT = 650
MIN_MATCH_COUNT = 4
ICECREAMTRUCK_DIMENSIONS = (20,15)
CROP_FACTOR = 0.2
MAX_ATTEMPTS = 5

class puzzleGUI:

    def __init__(self, master, video_source=1):
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
        self.getfile_button = tk.Button(master, text='Import Piece', command=self.get_piece_from_file)
        self.getglobal_button = tk.Button(master, text='Import Image', command=self.get_poster_image)

        self.camera_label = tk.Label(master, text='Camera Feed')
        self.camera_image = tk.Label(master, text='Live Camera Feed')
        self.capture_button = tk.Button(master, text='Capture Piece', command=self.snapshot)
        self.success_label = tk.Label(master, text=' ')
        self.matches_label = tk.Label(master, text=' ')
        self.rotation_label = tk.Label(master, text=' ')
        self.location_label = tk.Label(master, text=' ')
        self.attempts_label = tk.Label(master, text=' ')
        self.time_elapsed = tk.Label(master, text=' ')
        self.quit_button = tk.Button(master, text='Quit', command=master.quit)

        # Place GUI elements
        self.camera_label.grid(row=0, column=0, sticky="nsew")
        self.recent_label.grid(row=0, column=1, sticky="nsew")
        self.analysis_label.grid(row=0, column=2, sticky="nsew")

        self.camera_image.grid(row=1, column=0, rowspan=3, sticky="nsew")
        self.recent_image.grid(row=1, column=1, rowspan=3, sticky="nsew")
        self.analysis_image.grid(row=1, column=2, rowspan=3, sticky="nsew")

        self.capture_button.grid(row=4, column=0, sticky="nsew")
        self.getfile_button.grid(row=4, column=1, sticky="nsew")
        self.getglobal_button.grid(row=4, column=2, sticky="nsew")

        self.success_label.grid(row=5, column=0, sticky="nsew")
        self.matches_label.grid(row=6, column=0, sticky="nsew")
        self.attempts_label.grid(row=7, column=0, sticky="nsew")
        self.time_elapsed.grid(row=8, column=0, sticky="nsew")
        self.location_label.grid(row=6, column=1, sticky="nsew")
        self.rotation_label.grid(row=5, column=1, sticky="nsew")

        # Configure grid to resize with window
        for i in range(3):
            master.grid_columnconfigure(i, weight=1)
        for i in range(9):
            master.grid_rowconfigure(i, weight=1)

    def crop_image (self, image):
        factor = CROP_FACTOR
        (width, height, color) = image.shape
        return image[:,int(factor*height):int(height*(1-factor))]

    def snapshot(self):
        start = time.time()
        num_attempts = 0
        while num_attempts <= MAX_ATTEMPTS:
            # Get a frame from the video source
            ret, frame = self.vid.get_frame()
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = self.crop_image(frame)

            if ret:
                photo = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((int(CAMERA_WIDTH*(1-2*CROP_FACTOR)), CAMERA_HEIGHT)))
                self.recent_image.configure(image=photo)
                self.recent_image.photo = photo
                self.query_image = frame
                num_matches = self.analyze_piece()

            num_attempts += 1
            if num_matches >= MIN_MATCH_COUNT:
                break

        text = "Attempts: " + str(num_attempts)
        self.attempts_label.configure(text=text)
        # Measure the elapsed time
        end = time.time()
        text = str(round(end-start,3)) + " seconds"
        self.time_elapsed.configure(text=text)

    def get_poster_image(self):
        pil_photo, cv_photo = self.get_image_from_file()
        self.analysis_image.configure(image=pil_photo)
        self.analysis_image.photo = pil_photo
        self.train_image = cv_photo

        sift = cv2.SIFT_create()
        if self.train_image is not None:
            self.kp2, self.des2 = sift.detectAndCompute(self.train_image, None)
        # Get features of target image and analyze them at load rather than with every new piece

    def get_piece_from_file(self):
        pil_photo, cv_photo = self.get_image_from_file()
        self.recent_image.configure(image=pil_photo)
        self.recent_image.photo = pil_photo
        self.query_image = cv_photo
        self.analyze_piece()

    def get_image_from_file(self):
        name = tk.filedialog.askopenfilename(initialdir = '../puzzle-solver/input',
        filetypes = (('jpeg files','*.jpg'),('all files','*.*')))

        try: load = Image.open(name)
        except: return None, None

        resized = load.resize((CAMERA_WIDTH, CAMERA_HEIGHT))
        pil_photo = ImageTk.PhotoImage(resized)

        cv_photo = cv2.imread(name)

        return pil_photo, cv_photo

    def analyze_piece (self):
        # Read in query image, find features, and then find matches
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.query_image, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, self.des2, k=2)

        # Apply ratio test to only pick out the 'good' feature matches
        points = [];
        for m,n in matches:
            if m.distance < 0.70 * n.distance:  # May need to modify to get most # of pins
                points.append([int(self.kp2[m.trainIdx].pt[0]), int(self.kp2[m.trainIdx].pt[1])])
        points = np.float32(points)

        # Use k-means clustering to find where the most matches are
        k = 4
        if k > len(points):
            k = len(points)  # If less matches than 4
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        if len(points) > 1:
            compactness, label, center = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        good = []
        # Separate the data into clusters, find cluster with most elements, and get its centroid
        try:
            series = []
            for i in range(k):
                series.append(points[label.ravel()==i])
            mostElements = series[0]
            for s in series:
                if len(s) > len(mostElements):
                    mostElements = s
            (x, y) = (int(np.mean(mostElements[:,0])), int(np.mean(mostElements[:,1])))

            for m,n in matches:
                if m.distance < 0.7*n.distance:
                    if (int(self.kp2[m.trainIdx].pt[0]) in mostElements[:,0]):
                        good.append(m)
        except:
            pass # If not enough matches, then pass (don't send error to console)

        # Filter to keep just the good feature matches in the best cluster
        self.matches_label.configure(text="Matches: "+str(len(good)))

        # Rotate the piece back to the correct orientation
        img4 = self.train_image.copy()  # Copy of global image for feature-matching
        try: rows, cols, colors = self.query_image.shape
        except: rows, cols = self.query_image.shape # File input has 2 values, webcam has 3

        if len(good) >= MIN_MATCH_COUNT:
            self.success_label.configure(text="Success", fg="#0F0")

            src_pts = np.float32([kp1[m.queryIdx].pt for m in good ]).reshape(-1, 1, 2)
            dst_pts = np.float32([self.kp2[m.trainIdx].pt for m in good ]).reshape(-1, 1, 2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            # Estimate an affine transform
            Mprime = cv2.estimateAffinePartial2D(src_pts, dst_pts)
            degrees = math.degrees(math.atan2(Mprime[0][1][0], Mprime[0][0][0])) * (-1.0)

            turns = round(degrees/90.0) * 90
            if turns > 0:
                text = str(turns) + " counterclockwise"
            else:
                text = str(abs(turns)) + " clockwise"
            self.rotation_label.configure(text="Rotation: " + text)

            # Find the centroid of the cluster
            (ydim, xdim, colors) = self.train_image.shape
            (xx, yy) = ICECREAMTRUCK_DIMENSIONS # DORY_DIMENSIONS if 'dory' in puzzleName else MINION_DIMENSIONS
            (xdim, ydim) = (xdim/xx, ydim/yy)
            (xcoord, ycoord) = (int(math.floor(x/xdim)), int(math.floor(y/ydim)))
            self.location_label.configure(text="Location: ("+str(xcoord+1)+", "+str(ycoord+1)+")")  # Start from 0 is unnatural
            x = int(math.floor(x/xdim)*xdim + xdim/2)  # Discretize the centroid
            y = int(math.floor(y/ydim)*ydim + ydim/2)
            cv2.circle(img4, (x,y), 200, (0,0,255), 25)  # Draw centroid in blue for this specific piece output
        else:
            # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            self.success_label.configure(text="Failure", fg="#F00")
            self.rotation_label.configure(text="")
            self.location_label.configure(text="")

        # Resize and show on screen
        try: (x, y, c) = img4.shape
        except: (x, y) = img4.shape
        (newx, newy) = (int(math.floor(y*CAMERA_HEIGHT*1.0/x)), CAMERA_HEIGHT)
        resized = cv2.resize(img4, (newx, newy))
        img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        im_pil = ImageTk.PhotoImage(Image.fromarray(img))
        self.analysis_image.configure(image=im_pil)
        self.analysis_image.photo = im_pil

        return len(good) # Return the number of matches

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

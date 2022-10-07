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

    def __init__(self, master, video_source=0):
        self.master = master

        # Open video source (by default this will try to open the computer webcam)
        self.video_source = video_source
        self.cap=cv2.VideoCapture(0)
        self.vid = MyVideoCapture(self.video_source)

        self.setupGui(master)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.master.mainloop()

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
        self.capture_button = tk.Button(master, text='Capture', command=self.snapshot)
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


    def snapshot(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
             photo = ImageTk.PhotoImage(image = Image.fromarray(frame))
             self.camera_image.configure(image=photo)
             self.camera_image.photo = photo

         self.master.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Python Computer Vision Puzzle Solver')
    app = puzzleGUI(root)

import cv2
from PIL import Image, ImageTk
import tkinter as tk
import tkinter.filedialog

PADDING_X = 15
PADDING_Y = 10
IMAGE_WIDTH = 600
IMAGE_HEIGHT = 335
CAMERA_WIDTH = 533
CAMERA_HEIGHT = 400

class puzzleGUI:

    def __init__(self, master, video_source=0):
        self.master = master
        self.recent_image_file = None

        # Open video source (by default this will try to open the computer webcam)
        self.video_source = video_source
        self.cap=cv2.VideoCapture(0)
        self.vid = MyVideoCapture(self.video_source)

        self.setup_gui(master)

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()

        self.master.mainloop()

    def setup_gui (self, master):
        # Create GUI elements
        self.recent_label = tk.Label(master, text='Most Recent Piece')
        self.analysis_label = tk.Label(master, text='Target & Analysis')
        self.recent_image = tk.Label(master, text=' ')
        self.analysis_image = tk.Label(master, text=' ')
        self.getfile_button = tk.Button(master, text='Import', command=self.get_piece_from_file)
        self.getglobal_button = tk.Button(master, text='Import', command=self.get_poster_image)

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
            photo = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((CAMERA_WIDTH, CAMERA_HEIGHT)))
            self.recent_image.configure(image=photo)
            self.recent_image.photo = photo

    def get_poster_image(self):
        photo = self.get_image_from_file()
        self.analysis_image.configure(image=photo)
        self.analysis_image.photo = photo

    def get_piece_from_file(self):
        photo = self.get_image_from_file()
        self.recent_image.configure(image=photo)
        self.recent_image.photo = photo

    def get_image_from_file(self):
        name = tk.filedialog.askopenfilename(initialdir = '~/Coding',
        filetypes = (('jpeg files','*.jpg'),('all files','*.*')))

        load = Image.open(name)
        resized = load.resize((CAMERA_WIDTH, CAMERA_HEIGHT))
        photo = ImageTk.PhotoImage(resized)
        return photo

    def update(self):
         # Get a frame from the video source
         ret, frame = self.vid.get_frame()

         if ret:
            photo = ImageTk.PhotoImage(image = Image.fromarray(frame).resize((CAMERA_WIDTH, CAMERA_HEIGHT)))
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

import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detection with Tkinter")

        # Initialize the video capture
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            exit()

        # Load the pre-trained Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Create a label to display the video feed
        self.label = Label(self.root)
        self.label.pack()

        # Create a button to close the application
        self.close_button = Button(self.root, text="Close", command=self.close)
        self.close_button.pack()

        # Start the video feed
        self.update_video_feed()

    def update_video_feed(self):
        # Capture frame-by-frame
        ret, frame = self.cap.read()

        if not ret:
            print("Error: Could not read frame.")
            return

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Convert the frame t o RGB (OpenCV uses BGR by default)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert the frame to a PIL image
        pil_image = Image.fromarray(rgb_frame)

        # Convert the PIL image to an ImageTk object
        imgtk = ImageTk.PhotoImage(image=pil_image)

        # Update the label with the new frame
        self.label.imgtk = imgtk
        self.label.configure(image=imgtk)

        # Schedule the next update
        self.root.after(10, self.update_video_feed)

    def close(self):
        # Release the video capture
        self.cap.release()

        # Destroy the Tkinter window
        self.root.destroy()

# Create the Tkinter window
root = tk.Tk()

# Create the FaceDetectionApp object
app = FaceDetectionApp(root)

# Run the Tkinter main loop
root.mainloop()

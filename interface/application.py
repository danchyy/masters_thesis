import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
from tkinter import messagebox
import os
from utils.config import process_config
import cv2
from PIL import Image, ImageTk
from evaluator.model_predictor import ModelPredictor
from utils import constants
import json
import threading


class Application(tk.Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.path_to_model = None
        self.model_predictor = None
        width = 640
        height = 540
        x = (self.master.winfo_screenwidth() // 2) - (width // 2)
        y = (self.master.winfo_screenheight() // 2) - (height // 2)
        self.master.geometry('{}x{}+{}+{}'.format(640, 540, x, y))
        # self.configure_layout()
        # self.pack()
        # self.create_widgets()
        # layout all of the main containers
        self.master.grid_columnconfigure(0, weight=1)
        self.create_top_frame()

        self.create_center_frame()

        self.create_bottom_frame()
        print("Application is running!")

    def create_top_frame(self):
        self.top_frame = tk.Frame(master=self.master, width=640, height=100)
        self.top_frame.grid(row=0)

        self.select_model_button = tk.Button(master=self.top_frame)
        self.select_model_button["text"] = "Select Model"
        self.select_model_button["command"] = self.choose_model
        self.select_model_button.grid(row=0, column=0)

        self.selected_model_label = tk.Label(master=self.top_frame)
        self.selected_model_label["text"] = "Model not selected"
        self.selected_model_label.grid(row=1, column=0)

        self.select_video_button = tk.Button(master=self.top_frame)
        self.select_video_button["text"] = "Select video (json)"
        self.select_video_button["command"] = self.choose_video
        self.select_video_button.grid(row=0, column=1)

        self.selected_video_label = tk.Label(master=self.top_frame)
        self.selected_video_label["text"] = "Video not selected"
        self.selected_video_label.grid(row=1, column=1)

        self.predict_button = tk.Button(master=self.top_frame)
        self.predict_button["text"] = "Predict class of video"
        self.predict_button["state"] = "disabled"
        self.predict_button["command"] = self.predict
        # columnspan = 2
        self.predict_button.grid(row=2, columnspan=2)


    def create_center_frame(self):
        self.center_frame = tk.Frame(master=self.master, width=640, height=240)
        self.center_frame.grid(row=1)

        self.label = tk.Label(master=self.center_frame)
        self.label.grid(row=0)

    def create_bottom_frame(self):
        self.bottom_frame = tk.Frame(master=self.master, width=640, height=200)
        self.bottom_frame.grid(row=2)

        self.text = tk.Text(master=self.bottom_frame)
        self.text.grid(row=0)

    def choose_model(self):
        self.path_to_model = askdirectory(initialdir=os.path.join(constants.ROOT_FOLDER, constants.EXPORTED_MODELS_DIR))
        self.weights_path = os.path.join(self.path_to_model, "model.hdf5")
        config_path = os.path.join(self.path_to_model, "config.json")
        self.config = process_config(config_path)
        self.selected_model_label["text"] = self.path_to_model.split("/")[-1]

    def choose_video(self):
        if self.path_to_model is None:
            messagebox.showerror(title="Model not loaded", message="You have to load the model first!")
            return
        self.path_to_label_file = askopenfilename(initialdir=os.path.join(constants.UCF_101_EXTRACTED_FEATURES_TEST_1))
        with open(self.path_to_label_file) as f:
            json_data = json.load(f)
        self.features_path = json_data["mac_features_path"]
        self.video_path = json_data["video_path"]

        self.predict_button["state"] = "normal"
        display_name = self.features_path.split("/")[-1][:-3]
        self.selected_video_label["text"] = display_name

    def stream(self):
        success, frame = self.cap.read()
        if not success:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(img)
        self.label.configure(image=imgtk)
        self.label.img = imgtk
        # cv2.imshow(self.video_path.split("/")[-1], img)
        self.master.after(10, self.stream)

    def play(self):
        self.cap = cv2.VideoCapture(self.video_path)
        self.master.after(10, self.stream)

    def predict(self):
        if self.model_predictor is None:
            self.model_predictor = ModelPredictor(config=self.config, weights_path=self.weights_path)
        self.predictions = self.model_predictor.predict(self.features_path)

        video_name = self.video_path.split("/")[-2] + "/" + self.video_path.split("/")[-1]
        output_text = "Predictions for video: " + video_name + "\n"
        for output in self.predictions:
            output_text += output + "\n"
        self.text.delete('1.0', tk.END)
        self.text.insert(tk.END, output_text)

        self.cap = cv2.VideoCapture(self.video_path)
        self.master.after(10, self.stream)


    def configure_layout(self):
        self.columnconfigure(0, pad=3)

        self.rowconfigure(0, pad=5)


root = tk.Tk()
root.title("Action Recognition Demo")
root.lift()
root.call('wm', 'attributes', '.', '-topmost', True)
root.after_idle(root.call, 'wm', 'attributes', '.', '-topmost', False)
app = Application(master=root)
app.mainloop()

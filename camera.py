import cv2
import torch
import numpy as np
import io
from operator import truediv
import os
import json
from PIL import Image
import cv2

import torch
from flask import Flask, jsonify, url_for, render_template, request, redirect, Response
                
# RESULT_FOLDER = os.path.join('static')
# app.config['RESULT_FOLDER'] = RESULT_FOLDER

# finds the model inside your directory automatically - works only if there is one model
def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")
    
model_name = find_model()
model =torch.hub.load("WongKinYiu/yolov7", 'custom',model_name)

model.eval()  

class VideoCamera(object):
    def __init__(self):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        self.video = cv2.VideoCapture(0)
#        self.video = cv2.resize(self.video,(840,640))
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        #self.video = cv2.VideoCapture('video-test1.mp4')
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        success, image = self.video.read()
        results=model(image)
        a=np.squeeze(results.render())


        
       
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()
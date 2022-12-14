from flask import Flask, jsonify, url_for, render_template, request, redirect, Response
from camera import VideoCamera
import cv2
import os
from PIL import Image
import io
import torch

app = Flask(__name__)
RESULT_FOLDER = os.path.join('static')
app.config['RESULT_FOLDER'] = RESULT_FOLDER

def find_model():
    for f  in os.listdir():
        if f.endswith(".pt"):
            return f
    print("please place a model file in this directory!")

model_name = find_model()
model = torch.hub.load("WongKinYiu/yolov7", 'custom',model_name)

model.eval()

def get_prediction(img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]  # batched list of images
# Inference
    results = model(imgs, size=640)  # includes NMS
    return results

# Halaman awal
@app.route('/')
def index():
    return render_template('index.html')

# Halaman Ulpload Image File
@app.route('/image_file/', methods=['GET', 'POST'])
def image_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return
            
        img_bytes = file.read()
        results = get_prediction(img_bytes)
        results.save(save_dir='static')
        filename = 'image0.jpg'
        
        return render_template('result.html',result_image = filename,model_name = model_name)

    return render_template('image-file.html')    

# Halaman Video File
# @app.route('/multi/', methods=['GET', 'POST'])
# def multi():
#     if request.method == 'POST':
#         if 'file' not in request.files:
#             return redirect(request.url)
#         file = request.files.get('file')
#         if not file:
#             return  
#         return Response(gen(VideoCamera()),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

#     return render_template('multi.html') 

# Halaman Video Cam
@app.route('/video_cam/', methods=['GET', 'POST'])
def video_cam():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return  
        return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

    return render_template('video_cam.html')     

# Halaman Webcam
def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/webcam/')
def webcam():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)
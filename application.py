import re
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import os
import cv2

application = Flask(__name__) #Initialize the flask App


UPLOAD_FOLDER = "./upload"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = pickle.load(open('xray_model.pkl', 'rb'))

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(application.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

        image = cv2.imread('./upload/upload_chest.jpg') # read file 
        model.trainable=False
        ResizeImage = cv2.resize(image, dsize=(600, 600), interpolation=cv2.INTER_CUBIC)
        arr = np.expand_dims(ResizeImage,axis=0)
        prediction = model.predict(arr)
        classes =['COVID', 'Normal', 'Viral Pneumonia']
        res = classes[np.argmax(prediction)]
        return render_template('index.html', prediction_text=str(res) )

    if request.method == 'GET':
        return render_template('index.html')


if __name__ == "__main__":
    application.run(debug=True)

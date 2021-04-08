import os
import sys
import librosa
import keras
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, redirect
from flask import flash
from glob import glob
from werkzeug.utils import secure_filename


app=Flask(__name__)
app.secret_key = "secret key"

# Get current path
path = os.getcwd()

# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(audio_path, offset):
	y, sr = librosa.load(audio_path, offset=offset, duration=3)
	S = librosa.feature.melspectrogram(
	y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
	mfccs = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=40)
	# mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
	return mfccs


@app.route('/', methods=['GET','POST'])
def home():
    if request.method == 'POST':
        if "files" not in request.files:
            flash('No file part')
            return "File not chosen"
        
        files = request.files.getlist("files")

        uploads = request.form.get("folder")
        UPLOAD_FOLDER = os.path.join(path, uploads)

        # Make directory if uploads is not exists
        if not os.path.isdir(UPLOAD_FOLDER):
            os.mkdir(UPLOAD_FOLDER)

        app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File(s) successfully uploaded')  
        # print("UPLOAD_FOLDER:" ,UPLOAD_FOLDER)
        # test_folder_path = os.path.abspath(UPLOAD_FOLDER)
        # print("TEST FILES:" ,test_folder_path)
        model = load_model("heartbeat_classifier_DL.h5")
        entries = os.listdir(UPLOAD_FOLDER)
        results = []
        for one_file in entries:
            classify_file = one_file
            x_test = []
            x_test.append(extract_features(classify_file, 0.5))
            x_test = np.asarray(x_test)
            x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
            pred = model.predict(x_test, verbose=1)
            # print(pred)
            
            pred_class = model.predict_classes(x_test)
            print("pred_class", pred_class)
            if pred_class[0]:
                print(f"{classify_file}\nNormal heartbeat")
                results.append(1)
                print("confidence:", pred[0][1])
            else:
                print(f"{classify_file}\nAbnormal heartbeat")
                results.append(0)
                print("confidence:", pred[0][0])
    return render_template('home.html')
            



if __name__ == "__main__":
	# load model
    # model = load_model("trained_heartbeat_classifier.h5")
    app.run(debug=True, threaded=True)    
	# File to be classified
	# classify_file = "my_heartbeat.wav"
	
    

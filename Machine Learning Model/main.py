from flask import Flask, render_template, request, redirect
from flask import flash
import pandas as pd 
import librosa
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
from glob import glob
import cleaning
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import datetime
import os

app=Flask(__name__)

app.secret_key = "secret key"
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Get current path
path = os.getcwd()


# Allowed extension you can set your own
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

#load the pickle model
model = pickle.load(open("model.pkl", 'rb')) 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(y):
    mfcc=librosa.feature.mfcc(y,sr=44000,).T
    result = np.mean(mfcc)
    return result

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
        '''res=[]
        for i in range(0,len(y)):
            mfcc = extract_features(y[i])
            res.append(mfcc)
        prediction = model.predict(res)
        prediction = prediction[0]
        flash(prediction)'''
        #model = load_model('model.h5')
        #sound_dir = input()
        # print(sound_dir)
        '''
        data_dir = UPLOAD_FOLDER
        audio_files = glob(data_dir + '/*.wav')
        length = len(audio_files)'''
        sound_file, sample_rate = librosa.load('E:/Final Project/set_a/normal__201102201230.wav')
        #y = cleaning.clean(sound_file, sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=sound_file, sr=sample_rate, n_mfcc=40).T, axis=0)
        sound_feature = np.array(mfccs).reshape([-1,1])
        #sound_feature = sound_feature.reshape((1,40,1))

        prediction = model.predict(mfccs)
        #prediction = prediction[0][0]
        print(prediction)
        ans = ''

        if prediction == 0:
            ans = 'normal'
        else:
            ans = 'abnormal'
        print(ans)

#a0007 = abnormal
#a0001 = normal
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run(debug=True,threaded=True)    

# if __name__=='__main__':
#     app.run(debug=True, threaded=True)
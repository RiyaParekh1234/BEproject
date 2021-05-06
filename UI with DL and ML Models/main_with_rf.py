from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

#DL Model
import os, fnmatch
import sys
import librosa
import keras
import numpy as np
from keras.models import load_model
from flask import Flask, render_template, request, redirect
from flask import flash
from glob import glob
from werkzeug.utils import secure_filename
app = Flask(__name__)


#ML Model
import pandas as pd 
# import cleaning
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
import datetime

# Change this to your secret key (can be anything, it's for extra protection)
app.secret_key = 'your secret key'

# Enter your database connection details below
app.config['MYSQL_HOST'] = 'DESKTOP-6IBUKM8'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'root'
app.config['MYSQL_DB'] = 'pythonlogin'

# Intialize MySQL
mysql = MySQL(app)

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

def most_freq_dl(list):
    # print("Ok1")
    return np.bincount(list).argmax()
            
MAX_SOUND_CLIP_DURATION=12 #sec  
def audio_norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    data = (data-min_data)/(max_data-min_data+0.0001)
    return data-0.5

# get audio data without padding highest qualify audio
def load_file_data_without_change(folder,file_names, duration=3, sr=16000):
    input_length=sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file,res_type='kaiser_fast') 
            dur = librosa.get_duration(y=X, sr=sr)
            # extract normalized mfcc feature from data
            mfccs_ml = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0) 
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)
        feature = np.array(mfccs_ml).reshape([-1,1])
        data.append(feature)
    return data


# get audio data with a fix padding may also chop off some file
def load_file_data (folder,file_names, duration=12, sr=16000):
    input_length=sr*duration
    # function to load files and extract features
    # file_names = glob.glob(os.path.join(folder, '*.wav'))
    data = []
    for file_name in file_names:
        try:
            sound_file=folder+file_name
            print ("load file ",sound_file)
            # use kaiser_fast technique for faster extraction
            X, sr = librosa.load( sound_file, sr=sr, duration=duration,res_type='kaiser_fast') 
            dur = librosa.get_duration(y=X, sr=sr)
            # pad audio file same duration
            if (round(dur) < duration):
                print ("fixing audio lenght :", file_name)
                y = librosa.util.fix_length(X, input_length)                
            #normalized raw audio 
            # y = audio_norm(y)            
            # extract normalized mfcc feature from data
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sr, n_mfcc=40).T,axis=0)             
        except Exception as e:
            print("Error encountered while parsing file: ", file_name)        
        feature = np.array(mfccs).reshape([-1,1])
        data.append(feature)
    return data
@app.route('/pythonlogin/', methods=['GET', 'POST'])
def login():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s AND password = %s', (username, password,))
        # Fetch one record and return result
        account = cursor.fetchone()
        # If account exists in accounts table in out database
        if account:
            # Create session data, we can access this data in other routes
            session['loggedin'] = True
            session['id'] = account['id']
            session['username'] = account['username']
            # Redirect to home page
            # return 'Logged in successfully!'
            return redirect(url_for('home'))
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    # Show the login form with message (if any)
    return render_template('index.html', msg=msg)


# http://localhost:5000/python/logout - this will be the logout page
@app.route('/pythonlogin/logout')
def logout():
    # Remove session data, this will log the user out
   session.pop('loggedin', None)
   session.pop('id', None)
   session.pop('username', None)
   # Redirect to login page
   return redirect(url_for('login'))



# http://localhost:5000/pythinlogin/register - this will be the registration page, we need to use both GET and POST requests
@app.route('/pythonlogin/register', methods=['GET', 'POST'])
def register():
    # Output message if something goes wrong...
    msg = ''
    # Check if "username", "password" and "email" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form and 'email' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Check if account exists using MySQL
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE username = %s', (username,))
        account = cursor.fetchone()
        # If account exists show error and validation checks
        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not re.match(r'[A-Za-z0-9]+', username):
            msg = 'Username must contain only characters and numbers!'
        elif not username or not password or not email:
            msg = 'Please fill out the form!'
        else:
            # Account doesnt exists and the form data is valid, now insert new account into accounts table
            cursor.execute('INSERT INTO accounts VALUES (NULL, %s, %s, %s)', (username, password, email,))
            mysql.connection.commit()
            msg = 'You have successfully registered!'
    
    elif request.method == 'POST':
        # Form is empty... (no POST data)
        msg = 'Please fill out the form!'
    # Show registration form with message (if any)
    return render_template('register.html', msg=msg)


# http://localhost:5000/pythinlogin/home - this will be the home page, only accessible for loggedin users
@app.route('/pythonlogin/home', methods=['GET','POST'])
def home():
    # Check if user is loggedin
    if 'loggedin' in session:
        if request.method == 'POST':
            if "files" not in request.files:
                flash('No file part')
                return "File not chosen"
            
            files = request.files.getlist("files")

            uploads = request.form.get("folder")
            print("Foldername: ",uploads)
            UPLOAD_FOLDER = os.path.join(path, uploads)
            print("UPLOAD: ",UPLOAD_FOLDER)

            # Make directory if uploads is not exists
            if not os.path.isdir(UPLOAD_FOLDER):
                os.mkdir(UPLOAD_FOLDER)

            app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #flash('File(s) successfully uploaded') 
            model_name_dl = '<h3 style="color: #4a536e; font-size: 22px;">Deep Learning Model</h3>'
            flash(model_name_dl)
            print("UPLOAD_FOLDER:" ,UPLOAD_FOLDER)
            # test_folder_path = os.path.abspath(UPLOAD_FOLDER)
            # print("TEST FILES:" ,test_folder_path)
            model = load_model("heartbeat_classifier_DL.h5")
            entries = os.listdir(UPLOAD_FOLDER)
            results = []
            print("Enteries: ", entries)
            
            for one_file in entries:
                ########### New line added as compared to DL Model
                classify_file = os.path.join(UPLOAD_FOLDER, one_file)
                ######################
                x_test = []
                x_test.append(extract_features(classify_file, 0.5))
                x_test = np.asarray(x_test)
                x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)
                pred = model.predict(x_test, verbose=1)
                # print(pred)
                pred_class = model.predict_classes(x_test)
                print("pred_class", pred_class)
                if pred_class[0]:
                    print(f"{ one_file }\nNormal heartbeat")
                    #flash(f"{one_file} -> Normal heartbeat")
                    results.append(0)
                    print("confidence:", pred[0][1])
                else:
                    print(f"{one_file}\nAbnormal heartbeat")
                    #flash(f"{one_file} -> Abnormal heartbeat")
                    results.append(1)
                    print("confidence:", pred[0][0])
            print("results: ",results)

            if isinstance(results, list):
                # print("ok3") 
                predicted_dl = most_freq_dl(results)
                # print("ok2")
                print("predicted_dl: ",predicted_dl)

                if predicted_dl == 0:
                    #ans = 'normal'
                    flash(f"{one_file} -> Normal heartbeat")
                else:
                   # ans = 'abnormal'
                    flash(f"{one_file} -> Abnormal heartbeat")
            else:
                if results == 0:
                    #ans = 'normal'
                    flash(f"{one_file} -> Normal heartbeat")
                else:
                    #ans = 'abnormal'
                    flash(f"{one_file} -> Abnormal heartbeat")
            
            flash(predicted_dl)
            print("Machine learning starts...")
            # MACHINE LEARNING
            #load the pickle model
            ml_model = pickle.load(open("mod_RF_MachineLearning.pkl", 'rb')) 
            entries = os.listdir(UPLOAD_FOLDER)
            #results = []
            print("Enteries: ", entries)
            model_name_ml = '<h3 style="color: #4a536e; font-size: 22px;">Machine Learning Model</h3>'
            flash(model_name_ml)
            ml_upload_folder_path = UPLOAD_FOLDER + '/'
            test_files = fnmatch.filter(os.listdir(ml_upload_folder_path), '*.wav')
            test_sounds = load_file_data(folder=ml_upload_folder_path,file_names=test_files, duration=MAX_SOUND_CLIP_DURATION)
            test_labels = [-1 for items in test_sounds]
            testing_data = np.squeeze(test_sounds)
            prediction = ml_model.predict(testing_data)
            item = 0
            def most_freq(list):
                return np.bincount(list).argmax()
            if isinstance(prediction, np.ndarray): 
                predicted = most_freq(prediction)
                print(predicted)
                if predicted == 2:
                    #ans = 'normal'
                    flash(f"{one_file} -> Normal heartbeat")
                else:
                    #ans = 'abnormal'
                    flash(f"{one_file} -> Abnormal heartbeat")
            else:

                if prediction == 2:
                    #ans = 'normal'
                    flash(f"{one_file} -> Normal heartbeat")
                else:
                    #ans = 'abnormal'
                    flash(f"{one_file} -> Abnormal heartbeat")
                # print(ans)
                # item += 1
            print(prediction)
            #for one_file in entries:
                ########### New line added as compared to DL Model
                # classify_file = os.path.join(UPLOAD_FOLDER, one_file)
                # sound_file, sample_rate = librosa.load(classify_file)
                # #y = cleaning.clean(sound_file, sample_rate)
                # mfccs = np.mean(librosa.feature.mfcc(y=sound_file, sr=sample_rate, n_mfcc=40).T, axis=0)
                # sound_feature = np.array(mfccs).reshape([-1,1])
                # #sound_feature = sound_feature.reshape((1,40,1))
                
                
                
                #prediction = prediction[0][0]
                #print(prediction)
            #     #ans = ''
            #     if prediction[item] == 2:
            #        # ans = 'normal'
            #         # flash(f"{one_file} -> Normal heartbeat")
            #     else:
            #        # ans = 'abnormal'
            #         # flash(f"{one_file} -> Abnormal heartbeat")
            #    # print(ans)
             #   item += 1


        # User is loggedin show them the home page
        return render_template('home.html', username=session['username'])

    # User is not loggedin redirect to login page
    return redirect(url_for('login'))


# http://localhost:5000/pythinlogin/profile - this will be the profile page, only accessible for loggedin users
@app.route('/pythonlogin/profile')
def profile():
    # Check if user is loggedin
    if 'loggedin' in session:
        # We need all the account info for the user so we can display it on the profile page
        cursor = mysql.connection.cursor(MySQLdb.cursors.DictCursor)
        cursor.execute('SELECT * FROM accounts WHERE id = %s', (session['id'],))
        account = cursor.fetchone()
        # Show the profile page with account info
        return render_template('profile.html', account=account)
    # User is not loggedin redirect to login page
    return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True, threaded=True)    
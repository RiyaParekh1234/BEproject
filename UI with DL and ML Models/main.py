from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

#DL Model
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
                    flash(f"{one_file} -> Normal heartbeat")
                    results.append(1)
                    print("confidence:", pred[0][1])
                else:
                    print(f"{one_file}\nAbnormal heartbeat")
                    flash(f"{one_file} -> Abnormal heartbeat")
                    results.append(0)
                    print("confidence:", pred[0][0])

            #load the pickle model
            ml_model = pickle.load(open("ml_model_rf.pkl", 'rb')) 
            entries = os.listdir(UPLOAD_FOLDER)
            results = []
            print("Enteries: ", entries)
            model_name_ml = '<h3 style="color: #4a536e; font-size: 22px;">Machine Learning Model</h3>'
            flash(model_name_ml)
                
            for one_file in entries:
                ########### New line added as compared to DL Model
                classify_file = os.path.join(UPLOAD_FOLDER, one_file)
                sound_file, sample_rate = librosa.load(classify_file)
                #y = cleaning.clean(sound_file, sample_rate)
                mfccs = np.mean(librosa.feature.mfcc(y=sound_file, sr=sample_rate, n_mfcc=40).T, axis=0)
                sound_feature = np.array(mfccs).reshape([-1,1])
                sound_feature = sound_feature.reshape((1,40,1))
                

                prediction = ml_model.predict(sound_feature)
                #prediction = prediction[0][0]
                print(prediction)
                ans = ''
                if prediction == 0:
                    ans = 'normal'
                    flash(f"{one_file} -> Normal heartbeat")
                else:
                    ans = 'abnormal'
                    flash(f"{one_file} -> Abnormal heartbeat")
                print(ans)


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
# from flask import Flask, request, render_template
# from flaskext.mysql import MySQL

from flask import Flask, render_template, request, redirect, url_for, session
from flask_mysqldb import MySQL
import MySQLdb.cursors
import re

mysql = MySQL()

app = Flask(__name__)
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = 'root'
app.config['MYSQL_DATABASE_DB'] = 'doctor_data'
app.config['MYSQL_DATABASE_HOST'] = 'DESKTOP-6IBUKM8'
#app.config['MYSQL_DATABASE_PORT'] = 3306
#app.config['MYSQL_DATABASE_USER'] = 'root@localhost'
mysql.init_app(app)
@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/', methods=['POST'])
def auth():
    user = request.form['username']
    password = request.form['pass']

    cursor = mysql.connect().cursor()
    cursor.execute("select * from user where name='" + user + "' and password'" + password + "'")
    data = cursor.fetchone()

    if data is None:
        return "Username or Password is wrong!"

    else:
        return "Logged in Successfully!"

if __name__ == "__main__":
    app.run(debug=True)



from flask import Flask,render_template,request,redirect,url_for,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime
import joblib
import os

# weights = joblib.load("filename.pkl")
# baises = joblib.load("filename.pkl")

# Load the trained model
model_file_path = os.path.join(os.path.dirname(__file__), 'trained_model1.pkl')
# Load the trained model
pipe = pickle.load(open(model_file_path, 'rb'))

csv_file_path = os.path.join(os.path.dirname(__file__), 'Heart', 'merged.csv')

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
imputer.fit(df.drop(columns=['target']))  # Fit the imputer on the entire dataset, excluding the target column


# Function to preprocess input data
def preprocess_input_data(input_data):
    input_df = pd.DataFrame([input_data], columns=df.columns[:-1])  # Exclude the target column
    input_imputed = pd.DataFrame(imputer.transform(input_df), columns=input_df.columns)
    return input_imputed

app = Flask(__name__)
database_url = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_DATABASE_URI'] = database_url
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self, email, password, name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self, password):
        return bcrypt.checkpw(password.encode('utf-8'), self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()

        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')

    return render_template('login.html')


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')

    return render_template('register.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', user=user)

    return redirect('/login')


@app.route('/logout')
def logout():
    return redirect('/')

# Define a new model for storing user heart input data
class HeartInputData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    age = db.Column(db.Float)
    sex = db.Column(db.Float)
    cp = db.Column(db.Float)
    trestbps = db.Column(db.Float)
    chol = db.Column(db.Float)
    fbs = db.Column(db.Float)
    restecg = db.Column(db.Float)
    thalach = db.Column(db.Float)
    exang = db.Column(db.Float)
    oldpeak = db.Column(db.Float)
    slope = db.Column(db.Float)
    ca = db.Column(db.Float)
    thal = db.Column(db.Float)

    def __init__(self, user_id, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
        self.user_id = user_id
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalach = thalach
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal


with app.app_context():
    db.create_all()

# Modify the '/predict' route to store user input data into the database
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = float(request.form.get('age')) if request.form.get('age') else np.nan
    sex = float(request.form.get('sex')) if request.form.get('sex') else np.nan
    cp = float(request.form.get('cp')) if request.form.get('cp') else np.nan
    trestbps = float(request.form.get('trestbps')) if request.form.get('trestbps') else np.nan
    chol = float(request.form.get('chol')) if request.form.get('chol') else np.nan
    fbs = float(request.form.get('fbs')) if request.form.get('fbs') else np.nan
    restecg = float(request.form.get('restecg')) if request.form.get('restecg') else np.nan
    thalach = float(request.form.get('thalach')) if request.form.get('thalach') else np.nan
    exang = float(request.form.get('exang')) if request.form.get('exang') else np.nan
    oldpeak = float(request.form.get('oldpeak')) if request.form.get('oldpeak') else np.nan
    slope = float(request.form.get('slope')) if request.form.get('slope') else np.nan
    ca = float(request.form.get('ca')) if request.form.get('ca') else np.nan
    thal = float(request.form.get('thal')) if request.form.get('thal') else np.nan

    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    # Preprocess the input data
    input_data_imputed = preprocess_input_data(input_data)

    # Predict using the loaded model
    prediction = pipe.predict(input_data_imputed)

    # Store user input data in the database
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        if user:
            heart_input_data = HeartInputData(user_id=user.id, age=age, sex=sex, cp=cp, trestbps=trestbps,
                                              chol=chol, fbs=fbs, restecg=restecg, thalach=thalach,
                                              exang=exang, oldpeak=oldpeak, slope=slope, ca=ca, thal=thal)
            db.session.add(heart_input_data)
            db.session.commit()

    if prediction[0] == 0:
        result = "Not a Heart Patient"
    else:
        result = "A Heart Patient"

    return result

# Create a new route to display user input data
@app.route('/user_input_data')
def user_input_data():
    if 'email' in session:
        user = User.query.filter_by(email=session['email']).first()
        if user:
            user_heart_data = HeartInputData.query.filter_by(user_id=user.id).all()
            return render_template('user_input_data.html', user=user, user_heart_data=user_heart_data)

    return redirect('/login')


if __name__ == '__main__':
    app.run(debug=True)

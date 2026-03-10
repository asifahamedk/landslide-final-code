from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from werkzeug.security import generate_password_hash, check_password_hash
from collections import Counter
from ultralytics import YOLO
import os
import cv2
from werkzeug.utils import secure_filename
from ultralytics.utils.plotting import Annotator

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  # Set a secret key for session management
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['UPLOAD_FOLDER'] = 'uploads'

# Initialize the database and login manager
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# User Model for SQLAlchemy
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)

# Create the database tables if they do not exist
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Function to handle image bounding boxes
def boundingboxPredicted(results, model, image_path):
    output_folder = 'predictions'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image = cv2.imread(image_path)
    for r in results:
        annotator = Annotator(image)
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])

        img = annotator.result()
        output_image_path = os.path.join(output_folder, 'predictions.jpg')
        cv2.imwrite(output_image_path, img)
        
        # Return just the filename, not the full path
        return 'predictions.jpg'

# Function to run object detection
def run_object_detection(image_path):
    model_directory = r"C:\Users\91638\Desktop\LANDSLID_FINAL_CODE\WEBSITE"
    model_filename = "best.pt"
    model_path = os.path.join(model_directory, model_filename)

    infer = YOLO(model_path)
    result = infer.predict(image_path)
    item_counts = Counter(infer.names[int(c)] for r in result for c in r.boxes.cls)
    object_list = list(item_counts.keys())
    return object_list, result, infer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Use 'pbkdf2:sha256' as the hashing method
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        
        # Check if username already exists
        if User.query.filter_by(username=username).first():
            flash("Username already exists")
            return redirect(url_for('register'))
        
        new_user = User(username=username, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash("Registration successful. Please log in.")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash("Login failed. Check your username and password.")
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/process_image', methods=['POST'])
@login_required
def process_image():
    if 'image' not in request.files:
        return render_template('result.html', error="No file part")

    image_file = request.files['image']
    if image_file.filename == '':
        return render_template('result.html', error="No selected file")

    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    if '.' not in image_file.filename or image_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return render_template('result.html', error="Invalid file type")

    upload_dir = app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    object_list, result, infer = run_object_detection(image_path)
    predicted_image_path = boundingboxPredicted(result, infer, image_path)

    return render_template('result.html', image_filename=image_file.filename, object_list=object_list, predicted_image_path=predicted_image_path)

@app.route('/static/predictions/<filename>')
def predictions(filename):
    return send_from_directory('predictions', filename)

if __name__ == '__main__':
    app.run(debug=True)

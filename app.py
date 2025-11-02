from flask import Flask, render_template, request, send_from_directory
from model import get_emotion
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)
            emotion = get_emotion(path)

    return render_template('index.html', emotion=emotion, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

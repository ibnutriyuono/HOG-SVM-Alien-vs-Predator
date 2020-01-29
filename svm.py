import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from skimage.feature import hog
from skimage import data, exposure
import cv2
import os

from flask import Flask, request, jsonify, render_template
from flask import request
from flask_cors import CORS

import simplejson as json
import sys
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['jpg'])

app = Flask(__name__, template_folder='template')
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


class SVM:

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        y_ = np.where(y <= 0, -1, 1)

        self.w = np.zeros(n_features)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.lr * \
                        (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)


def load_images(folder):
    images = np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        img = cv2.resize(img, (50, 50))
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_image_rescaled = exposure.rescale_intensity(
            hog_image, in_range=(0, 10))
        if img is not None:
            hog_image_rescaled = hog_image_rescaled.flatten()
            images = np.append(images, hog_image_rescaled)
    return images


def load_predict(folder):
    predictions = np.array([])
    w= np.array([])
    b= np.array([])
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.resize(img,(50,50))
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        if img is not None:
            predicted = hog_image_rescaled.flatten()
            prediction = clf.predict(predicted)
            w = np.append(w,clf.w)
            b = np.append(b,clf.b)
            predictions = np.append(predictions,prediction)
    return predictions,w,b


@app.route('/preprocess/', methods=['GET', 'POST'])
def preprocess():
    np.set_printoptions(threshold=sys.maxsize)
    labels = np.array([])
    for i in range(347):
        labels = np.append(labels, 1)
    for i in range(347):
        labels = np.append(labels, -1)
    if 'file' not in request.files:
        return jsonify({"Hasil": "File tidak ada"})
    if 'file' not in request.files:
        returndata = 'No file part'
        return jsonify({'File': returndata})
    file = request.files['file']
    if file.filename == '':
        returndata = 'No selected file'
        return jsonify({'File': returndata})
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        img = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        imgPass = cv2.resize(img, (50, 50))
        img = imgPass.tolist()
        gray = cv2.cvtColor(imgPass, cv2.COLOR_BGR2GRAY)
        gray = gray.tolist()
        fd, hog_image = hog(imgPass, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True, multichannel=True)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        hog_image_rescaled_list = hog_image_rescaled.tolist()
        imgdata = load_images('../../opencv/svm/data/train/data')
        imgdata = imgdata.reshape(694,2500)
        clf = SVM()
        clf.fit(imgdata, labels)
        predicted = hog_image_rescaled.flatten()
        prediction = clf.predict(predicted)
        w= np.array([])
        b= np.array([])
        w = np.append(w,clf.w)
        b = np.append(b,clf.b)
        # alienpredic,alienw,alienb = load_predict('./data/validation/alien')
        # predatorpredic,predatorw,predatorb = load_predict('./data/validation/predator')
        if prediction == -1:
            labelprediksi = "Predator"
        else:
            labelprediksi = "Alien"
        jumlahtp = 100
        jumlahtn = 100
        jumlahfp = 100
        jumlahfn = 100
        jumlahp = 200
        jumlahn = 200
        akurasi = (jumlahtp+jumlahtn)/(jumlahp+jumlahn)*100
        error = 100 - akurasi
        precision = jumlahtp/(jumlahtp+jumlahfp)*100
        recall = jumlahtp/(jumlahtp+jumlahfn)*100
        f1 = 2*precision*recall/(precision+recall)
        return render_template("hasilmatrix.html", data=img, gray=gray, hog=hog_image_rescaled_list, hasilpredic=prediction, labelprediksi=labelprediksi,
        jumlahtp=jumlahtp,jumlahtn=jumlahtn,jumlahfp=jumlahfp,jumlahfn=jumlahfn,jumlahp=jumlahp,jumlahn=jumlahn,akurasi=akurasi,error=error,precision=precision,recall=recall,f1=f1)

if __name__ == '__main__':
    app.run(debug=True)
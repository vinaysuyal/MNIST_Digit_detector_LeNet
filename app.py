#Maching Learning Dependencies
from Models import LeNet
import torch
#import torchvision
#import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import torch.optim as optim
import pickle
from PIL import Image
#from matplotlib import cm
from skimage.transform import rescale, resize
from sklearn.impute import SimpleImputer

import os
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

UPLOAD_FOLDER = '/static'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(image_name):

    file_name = 'digit_detector.pkl'
    m1 = pickle.load(open(file_name,'rb'))

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    #trans = transforms.Compose([transforms.RandomResizedCrop(28)])


    x = Image.open(image_name)
    y = asarray(x)

    y = imp.fit_transform(y[:,:,0])

    x = resize(y,(28,28),anti_aliasing=True)

  
    x = torch.from_numpy(x)
    x = x.unsqueeze(0) # adding channel
    x = x.unsqueeze(0) # adding batch_size


    x = x.type(torch.FloatTensor)
    output = m1(x)
    print(output)
    a,b = torch.max(output.data, 1)
    #print("The digit in the image appears to be",b.item())
    return b.item()





@app.route("/")
def index():
    return "Welcome to the Website"

@app.route("/detect_digit_mnist", methods = ['POST','GET'])
def detect_digit_MNIST():
    if request.method == "POST":
        #image = request.form.get('img')
        image = request.files['img']
        filename = secure_filename(image.filename)
        image.save(filename)
        print(filename)
        pred = predict(filename)
        return jsonify({"Predicted Output":pred})
    
    return render_template('predict.html')

app.run(port = 8000)

if __name__ == "__main__":
    app.run(debug=True)
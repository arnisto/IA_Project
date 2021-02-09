from flask import Flask, render_template , request
import pickle
import numpy as np
#model = pickle.load(open('first_model_for_predecting_sales.sav','rb'))
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import time
import sys


UPLOAD_FOLDER = 'images_uploaded'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# load json and create model

new_model = tf.keras.models.load_model('models.h5')

def image_resize(img, size = (None, None), ratio=3):
    if size[0] is None:
        resize_ratio = ratio
        resize_height = int(img.shape[0]/resize_ratio)
        resize_width = int(img.shape[1]/resize_ratio)
        print(f"height: {resize_height}, width: {resize_width}")
    else:
        resize_height = size[0]
        resize_width = size[1]

    img_resize = tf.image.resize(img, [resize_height,resize_width]).numpy()
    img_resize = img_resize.astype(np.uint8)
    return(img_resize)

def img_to_data(file) :
    img_height = 227
    img_width = 341   
    train_resized_input = [] 
    image_input = plt.imread(file)
    train_resized_input.append(image_resize(image_input, (img_height, img_width)))
    x_train_input = np.ndarray(shape = (len(train_resized_input), img_height, img_width, 3), dtype=np.float32)
    for i in range(len(train_resized_input)):
        x_train_input[i] = img_to_array(train_resized_input[i])
        x_train_input = x_train_input/255
    print(x_train_input.shape) 
    return x_train_input










app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def man():
    return render_template('index.html')

@app.route('/result',methods=['GET','POST'])
def home():
      if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        for filename in os.listdir('static/'):
            if filename.startswith('gg'): 
                os.remove('static/' + filename)
        # Save the file to ./uploads
        basepath = os.path.join(os.path.dirname(__file__))
        #basepath = 'C:/Users/lamjed/Desktop/dev_app_mobil/flask/projet_ia'
        x="gguploaded"+str(time.time())+".jpg"
        file_path = os.path.join(
            basepath, 'static', secure_filename(x))
        f.save(file_path)
        #make some data
        data = img_to_data(file_path)
     
        new_data = new_model.predict(data)
        list_of_data = list(new_data)
        print(list_of_data[0])
        return render_template('result.html', myData= list_of_data[0])

if __name__ == "__main__":
    app.debug = True
    app.run()
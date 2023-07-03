from flask import Flask,render_template,request


app = Flask(__name__)

@app.route('/',methods=['GET'])
def hello_world():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import pathlib
    import os
    import glob as gb
    import cv2 as cv
    import PIL
    import seaborn as sns
    import tensorflow
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from tensorflow.keras.callbacks import EarlyStopping ,ReduceLROnPlateau 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D , Dense , Dropout , Flatten , MaxPooling2D , BatchNormalization ,experimental
    from keras.utils.np_utils import to_categorical
    from keras.applications.vgg16 import VGG16
    from keras.applications.vgg19 import VGG19
    from tensorflow import keras
    from keras.models import Model
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    import seaborn as sns
    return render_template('index.html')

import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pickle
import cv2 as cv
# @app.route('/', methods=['POST'])
# def predict():
#     imagefile = request.files['imagefile']
#     image_dir = "./images/"

#     # Create the "images" directory if it doesn't exist
#     if not os.path.exists(image_dir):
#         os.makedirs(image_dir)

#     image_path = os.path.join(image_dir, imagefile.filename)
#     imagefile.save(image_path)

#     # Load and preprocess the image
#     img = Image.open(image_path)
#     img = img.resize((224, 224))
#     img_array = img_to_array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     pickle_file_path2 = 'C:/Users/Debashish/Downloads/fruit_detection/etc.pkl'
#     with open(pickle_file_path2, 'rb') as file:
#         model3 = pickle.load(file)
#     # Perform the prediction using model3
#     prediction = model3.predict(img_array)
#     predicted_class = np.argmax(prediction)

#     # Define the class labels
#     class_labels = {0: 'Healthy', 1: 'BlackSpot', 2: 'Anthracnose'}

#     # Get the predicted class label
#     predicted_label = class_labels[predicted_class]

#     # Update the predict function to include the prediction result in the rendered template
#     return render_template('index.html', prediction=predicted_label)

@app.route('/predict', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_dir = "./images/"

    # Create the "images" directory if it doesn't exist
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    image_path = os.path.join(image_dir, imagefile.filename)
    imagefile.save(image_path)

    # Load and preprocess the image
    train_images=[]
    image_raw=cv.imread(image_path)
    image=cv.cvtColor(image_raw,cv.COLOR_BGR2RGB)
    resize_image=cv.resize(image,(224,224))
    train_images.append(list(resize_image))
    X_train=np.array(train_images)
    
    

    pickle_file_path0 = 'C:/Users/Debashish/Downloads/fruit_detection (1)/fruit_detection/model_FE_16.pkl'
    with open(pickle_file_path0, 'rb') as file0:
        model0 = pickle.load(file0)

    pickle_file_path1 = 'C:/Users/Debashish/Downloads/fruit_detection (1)/fruit_detection/model_FE_19.pkl'
    with open(pickle_file_path1, 'rb') as file1:
        model1 = pickle.load(file1)



    pickle_file_path2 = 'C:/Users/Debashish/Downloads/fruit_detection (1)/fruit_detection/etc.pkl'
    with open(pickle_file_path2, 'rb') as file:
        model3 = pickle.load(file)

    feature_16=model0.predict(X_train)
    feature_19=model1.predict(X_train)
    final_feature=np.hstack((feature_16,feature_19))

    # Reshape img_array to match the expected input shape of model3
    #img_array_reshaped = img_array.reshape(img_array.shape[0], -1)[:,:1024]

    # Perform the prediction using model3
    prediction = model3.predict(final_feature)
    predicted_class = np.argmax(prediction)

    # Define the class labels
    class_labels = {0: 'Healthy', 1: 'BlackSpot', 2: 'Anthracnose'}

    # Get the predicted class label
    predicted_label = class_labels[predicted_class]

    # Update the predict function to include the prediction result in the rendered template
    return render_template('prediction.html', prediction=predicted_label)

if __name__=='__main__':
    app.run(port=8080,debug=True)

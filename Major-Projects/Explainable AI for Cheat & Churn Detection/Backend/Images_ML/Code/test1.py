import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import shutil

# -------------------
# USER INPUT
# -------------------
fileName = input("Enter image filename (inside test/ folder): ")

# -------------------
# CLEAN FOLDER & COPY IMAGE
# -------------------
dirPath = "static/images"
fileList = os.listdir(dirPath)

# Remove existing images
for file in fileList:
    os.remove(os.path.join(dirPath, file))

# Copy new image
shutil.copy(os.path.join("test", fileName), dirPath)
print("Image copied to static/images")

verify_dir = "static/images"

# -------------------
# IMAGE PROCESSING
# -------------------
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'cheating-{}-{}.model'.format(LR, '2conv-basic')

def process_verify_data():
    verifying_data = []
    for img in os.listdir(verify_dir):
        path = os.path.join(verify_dir, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        verifying_data.append([np.array(img), img_num])
    np.save('verify_data.npy', verifying_data)
    return verifying_data

verify_data = process_verify_data()

# -------------------
# BUILD MODEL
# -------------------
tf.compat.v1.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, 
                     loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)

# Load model
if os.path.exists(MODEL_NAME + '.meta'):
    model.load(MODEL_NAME)
    print("Model loaded!")
else:
    print("Model not found!")

# -------------------
# PREDICTION
# -------------------
for num, data in enumerate(verify_data):
    img_num = data[1]
    img_data = data[0]

    data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)

    model_out = model.predict([data])[0]
    print("Model raw output:", model_out)

    if np.argmax(model_out) == 0:
        label = "CHEATING"
        accuracy = model_out[0] * 100
        print(f"\nPCHEATING {accuracy:.2f}%")
    else:
        label = "NORMAL"
        accuracy = model_out[1] * 100
        print(f"\nPREDICTION: NORMAL brain with accuracy {accuracy:.2f}%")

    # Display image
    plt.imshow(cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB))
    plt.title(f"{label} ({accuracy:.2f}%)")
    plt.axis('off')
    plt.show()

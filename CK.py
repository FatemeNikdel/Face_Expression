#import mtcnn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"
import cv2
import tensorflow
from mtcnn import MTCNN
import glob
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import pickle
from sklearn.model_selection import train_test_split
import time
import matplotlib.pyplot as plt

######################################## Loading Data & Preprocess #########################################
path = r'CK+\\*\\*'
data  = []
label = []
for i, path in enumerate(glob.glob(path)):
    #print(path)
    img = cv2.imread(path)
    img = cv2.resize(img, (64, 64))/ 255.0
    data.append(img)
    labels = path.split("\\")[-2]
    label.append(labels)
    if i % 100 == 0:
        print(f"[Info] the {i}th preprocessed!")   

lb = LabelBinarizer()
label = lb.fit_transform(label)
data = np.array(data)
label = np.array(label)
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2)

##################################### Data Augmentation ####################################################
aug = tensorflow.keras.preprocessing.image.ImageDataGenerator(
                        rotation_range = 20,
                        width_shift_range = 0.1,
                        height_shift_range = 0.1,
                        shear_range = 0.2,
                        zoom_range = 0.2,
                        horizontal_flip = True,
                        fill_mode = "nearest"
                        )
                        
##################################### Defining Network #####################################################
CNN_net = tensorflow.keras.models.Sequential([
                            tensorflow.keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = (64,64,3)),
                            tensorflow.keras.layers.BatchNormalization(),
                            tensorflow.keras.layers.MaxPool2D(),

                            tensorflow.keras.layers.Conv2D(64, (3,3), activation = "relu"),
                            tensorflow.keras.layers.BatchNormalization(),
                            tensorflow.keras.layers.MaxPool2D(),
                            
                            tensorflow.keras.layers.Flatten(),
                            tensorflow.keras.layers.Dense(32, activation = "relu"),
                            tensorflow.keras.layers.BatchNormalization(),
                            tensorflow.keras.layers.Dense(7, activation = "softmax"),
                            ])
opt = tensorflow.keras.optimizers.Adam(learning_rate = 0.01, decay = 0.0008)
CNN_net.compile(
                optimizer = opt,
                metrics = ['accuracy'],
                loss = 'categorical_crossentropy'
                )
#print(X_train.shape)
#print(y_train.shape)
begin = time.time()
CNN_net.fit(
            aug.flow(X_train, y_train, batch_size = 64),
            steps_per_epoch = len(X_train)//64,
            validation_data = (X_test, y_test),
            epochs = 30
            )
end = time.time()
print("Total time of running Net is {}".format(end-begin))












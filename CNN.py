import numpy as np
import glob 
import cv2 
import tensorflow as tf
from tensorflow import keras
import os

training_fruit_img = []
training_label = []
test_fruit_img = []
test_label = []

def getImagesAndLabels(path: str, resizeDims: int):
    """
    For all images in a directory, returns resized images in an array 
    with a corresponding label based on their subdirectory.
    """
    images = []
    labels = []
    for dir_path in glob.glob(path):
        img_label = dir_path.split("\\")[-1]
        for img_path in glob.glob(os.path.join(dir_path, "*.jpg")):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (resizeDims, resizeDims))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(img_label)
    return images, labels

training_fruit_img, training_label = getImagesAndLabels("fruits-360-original-size/Training/*", 64)
test_fruit_img, test_label = getImagesAndLabels("fruits-360-original-size/Validation/*", 64)
training_fruit_img = np.array(training_fruit_img)
training_label = np.array(training_label)
test_fruit_img = np.array(test_fruit_img)
test_label = np.array(test_label)

label_to_id = {v : k for k, v in enumerate(np.unique(training_label))}
id_to_label = {v : k for k, v in label_to_id.items()}

training_label_id = np.array([label_to_id[i] for i in training_label])
test_label_id = np.array([label_to_id[i] for i in test_label])

training_fruit_img = training_fruit_img / 255.0
test_fruit_img = test_fruit_img / 255.0

model = keras.Sequential()
model.add(keras.layers.Conv2D(16, (3, 3), input_shape = (64, 64, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(32, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Conv2D(64, (3, 3), padding = "same", activation = "relu"))
model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(256, activation = "relu"))
model.add(keras.layers.Dense(75, activation = "softmax"))

model.compile(loss = "sparse_categorical_crossentropy", optimizer = keras.optimizers.Adamax(), metrics = ['accuracy'])
tensorboard = keras.callbacks.TensorBoard(log_dir = "./Graph", histogram_freq = 0, write_graph = True, write_images = True)
model.fit(training_fruit_img, training_label_id, batch_size = 128, epochs = 10, callbacks = [tensorboard])

score = model.evaluate(test_fruit_img, test_label_id, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

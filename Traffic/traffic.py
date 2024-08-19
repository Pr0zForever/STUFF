import cv2
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.model_selection import train_test_split

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Plot Data based on training data and test data
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    
    images = []
    labels = []
    count = 0
    for ifold in os.listdir(data_dir):
        for im in os.listdir(os.path.join(data_dir, ifold)):
            im = cv2.cvtColor(cv2.imread(os.path.join(data_dir,ifold,im)), cv2.COLOR_BGR2RGB)
            images.append(cv2.resize(im,(IMG_WIDTH,IMG_HEIGHT)))
            labels.append(ifold)
        print(count)
        count+=1
    return images,labels
    


def get_model():
    
    model = models.Sequential()
    model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(30, 30, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(NUM_CATEGORIES))
    model.compile(optimizer='adam',loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    return model

if __name__ == "__main__":
    main()

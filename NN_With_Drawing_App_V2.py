# Import necessary libraries
from keras.models import Sequential
from keras.layers import LSTM, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import numpy as np
import np_utils

import os
import requests
import urllib.parse

import copy


# Function to load and preprocess the data
def load_data(files, directory= 'D:/School/F23/ML/ML Project/data/'):
    data = []
    labels = []
    # Load the data
    for idx, file in enumerate(files):
        file_path = directory + file
        doodles = np.load(file_path)
        # Here we might want to limit the number of doodles per class to avoid class imbalance
        doodles = doodles[:1000]  # Assuming we are using 10,000 samples per class
        data.append(doodles)
        labels.extend([idx] * len(doodles))
    # Flatten the data and convert to float32 for precision
    data = np.concatenate(data).astype('float32')
    # Normalize the data
    data /= 255.0
    # Reshape for NN input
    data = data.reshape(-1, 28, 28, 1)
    return data, np.array(labels)
 
# Create a simple CNN model
def create_cnn_model(num_classes):
    model = Sequential()

    # Convolutional layers
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Fully connected layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the CNN model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# Create a simple ANN model
def create_ann_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
# Create a simple RNN model
def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

## ---------- INITIALIZE DATA ---------- ##

# Add new foods to the dictionary and new file to the files
# Make sure to download the numpy bitmap to the data folder

labels_list = ['Apple', 'Banana', 'Broccoli', 'Grapes', 'Pineapple']

labels_dict = {}
files = []
num_classes = len(labels_list)

for i in range(num_classes):

    labels_dict[i] = (labels_list[i])
    
    file = labels_list[i].lower() + '.npy'
    files.append(file)
    
# TRIED TO AUTOMATICALLY DOWNLOAD NEW DATASET FROM ONLINE, FAILED

#save_path = 'D:/School/F23/ML/ML Project/data/'
#category = "ice cream"

# Encode the category for the URL
#encoded_category = urllib.parse.quote(category)

#base_url = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/"
#url = f"{base_url}/{encoded_category}.npy"
#response = requests.get(url)

#if response.status_code == 200:
    # Save the downloaded data as a .npy file
#    file_path = os.path.join(save_path, f"{encoded_category}.npy")
#    with open(file_path, 'wb') as f:
#        f.write(response.content)
#    print(f"Data for '{category}' downloaded and saved to {file_path}")
#else:
#    print(f"Failed to download data for '{category}'. Status code: {response.status_code}")
    
    
print("Loading data from ", files, "...")
data, labels = load_data(files)

# Split the data into training and test sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)

# Convert class vectors to binary class matrices (one-hot encoding)
print("Converting labels into categories...")
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Pre shape: ", x_train.shape)

## ---------- CONVOLUTIONAL NEURAL NETWORK ---------- ##

print("Creating the Convolutional Neural Network...")
cnn_model = create_cnn_model(num_classes=len(files))

print("Training the Convolutional Neural Network...")
cnn_model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

print("Testing the Convolutional Neural Network...")
score = cnn_model.evaluate(x_test, y_test, verbose=0)
print('CNN Test loss:', score[0])
print('CNN Test accuracy:', score[1])

print("\nConvolutional Neural Network Successfully Created!\n")

## ---------- ARTIFICIAL NEURAL NETWORK ---------- ##

# Flatten the images for ANN input
x_train_ann = copy.deepcopy(x_train)
x_test_ann = copy.deepcopy(x_test)

print("ANN shape: ", x_train.shape)

x_train_ann = x_train_ann.reshape(x_train.shape[0], -1)
x_test_ann = x_test_ann.reshape(x_test.shape[0], -1)

print("Creating the Artificial Neural Network...")
ann_model = create_ann_model((x_train_ann.shape[1],), len(files))

print("Training the Artificial Neural Network...")
ann_model.fit(x_train_ann, y_train, batch_size=32, epochs=10, validation_data=(x_test_ann, y_test))

print("Testing the Artificial Neural Network...")
score = ann_model.evaluate(x_test_ann, y_test, verbose=0)
print('ANN Test loss:', score[0])
print('ANN Test accuracy:', score[1])

print("\nArtificial Neural Network Successfully Created!\n")

## ---------- RECURRENT NEURAL NETWORK ---------- ##

# Flatten the images for ANN input
x_train_rnn = copy.deepcopy(x_train)
x_test_rnn = copy.deepcopy(x_test)

print("RNN shape: ", x_train.shape)

# Reshaping to (samples, time steps, features)
x_train_rnn = x_train_rnn.reshape(x_train.shape[0], x_train.shape[1], -1)  
x_test_rnn = x_test_rnn.reshape(x_test.shape[0], x_test.shape[1], -1)

# Create and compile the RNN model
print("Creating the Recurrent Neural Network...")
rnn_model = create_rnn_model(x_train_rnn.shape[1:], len(files))

# Train the model

print("Training the Recurrent Neural Network...")
rnn_model.fit(x_train_rnn, y_train, batch_size=32, epochs=10, validation_data=(x_test_rnn, y_test))

# Evaluate the model
print("Testing the Recurrent Neural Network...")
score = rnn_model.evaluate(x_test_rnn, y_test, verbose=0)
print('RNN Test loss:', score[0])
print('RNN Test accuracy:', score[1])

print("\nRecurrent Neural Network Successfully created!\n")

## ---------- BUILDING DRAWING APP ---------- ##

import tkinter as tk
import matplotlib.pyplot as plt

from tkinter import messagebox
from PIL import Image, ImageDraw
from random import choice

class DrawingApp:
    def __init__(self, root):
    
        target_class = choice(list(labels_dict.values()))
        
        self.root = root
        self.root.title(f"Drawing App - Draw a {target_class}")
        
        self.numpyRay = np.zeros(28*28)

        self.canvas = tk.Canvas(root, bg="black", width=400, height=400)
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)

        self.guess_button = tk.Button(root, text="Guess Drawing", command=self.guess_drawing)
        self.guess_button.pack()

        self.reset_button = tk.Button(root, text="Reset", command=self.reset_drawing)
        self.reset_button.pack()
       
        self.target_class = target_class
        self.last_x = None
        self.last_y = None
        self.brush_size = 15  # Initial brush size
        self.image = Image.new("RGB", (400, 400), "black")
        self.draw = ImageDraw.Draw(self.image)
        

    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y

    def draw(self, event):
        if self.target_class:
            x, y = event.x, event.y
            color = "white"
            self.canvas.create_line(
                (self.last_x, self.last_y, x, y),
                fill=color,
                width=self.brush_size
            )
            self.draw.line(
                (self.last_x, self.last_y, x, y),
                fill=color,
                width=self.brush_size
            )
            self.last_x = x
            self.last_y = y
         

    def predict_image(self, image):
    
        # Preprocess the image for the CNN model
        image_cnn = image.astype('float32') / 255.0
        image_cnn = image_cnn.reshape(1, 28, 28, 1)

        # Predict with the CNN model
        pred_cnn = cnn_model.predict(image_cnn)
        predicted_class_cnn = np.argmax(pred_cnn)
        confidence_cnn = np.max(pred_cnn)

        # Preprocess the image for the ANN model
        image_ann = image.flatten().reshape(1, -1) / 255.0

        # Predict with the ANN model
        pred_ann = ann_model.predict(image_ann)
        predicted_class_ann = np.argmax(pred_ann)
        confidence_ann = np.max(pred_ann)

        # Preprocess the image for the RNN model
        image_rnn = image.reshape(1, 28, -1) / 255.0

        # Predict with the RNN model
        pred_rnn = rnn_model.predict(image_rnn)
        predicted_class_rnn = np.argmax(pred_rnn)
        confidence_rnn = np.max(pred_rnn)

        return {
            'CNN': (labels_dict[predicted_class_cnn], confidence_cnn),
            'ANN': (labels_dict[predicted_class_ann], confidence_ann),
            'RNN': (labels_dict[predicted_class_rnn], confidence_rnn)
        }

    def guess_drawing(self):

        # Convert the image to a 28x28 numpy array
        resized_image = self.image.resize((28, 28), Image.LANCZOS)
        image_array = np.array(resized_image)
        image_array_bw = np.mean(image_array, axis=2).astype(np.uint8)
        #np.save(os.path.join(folder_path, folder_name + ".npy"), image_array_bw)

        # Display the resized numpy array using matplotlib
        #plt.imshow(image_array_bw, cmap='gray')
        #plt.title("Resized Image")
        #plt.show()
        
        results = self.predict_image(image_array_bw)
        maxConf = -1e7
        maxLab = "None"
        modNam = "None"
        for model_name, result in results.items():
            print(f"{model_name} Prediction: Class - {result[0]}, Confidence - {result[1]:.2f}")
            if(result[1] > maxConf):
                maxConf = result[1]
                maxLab = result[0]
                modNam = model_name
        
        correct = messagebox.askyesno("Guess:", f"Is this a {maxLab}?\n{modNam} Prediction: Class - {maxLab}, Confidence - {maxConf:.2f}")
        
        if correct:
            print("Yay! I was right! :)")
        else:
            print("Oh no, i was wrong... :(")
            
        
        self.reset_drawing()


    def reset_drawing(self):
        self.canvas.delete("all")  # Clear the canvas
        self.image = Image.new("RGB", (400, 400), "black")
        self.draw = ImageDraw.Draw(self.image)
        target_class = choice(list(labels_dict.values()))
        self.root.title(f"Drawing App - Draw a {target_class}")
 

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()



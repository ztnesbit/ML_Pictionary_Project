## ---------- CREATING CONVOLUTIONAL NEURAL NETWORK ---------- ##

# Import necessary libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import np_utils
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np

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

# Define the CNN model architecture
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

# Now adjust your main code to load the data
files = ['apple.npy', 'banana.npy', 'grapes.npy', 'pineapple.npy']
print("Loading data from ", files, "...")
data, labels = load_data(files)

# Split the data into training and test sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)

# Convert labels to categorical
print("Converting labels into categories...")
y_train = to_categorical(y_train, num_classes=len(files))
y_test = to_categorical(y_test, num_classes=len(files))

# Create and compile the model
print("Creating the Convolutional Neural Network...")
model = create_cnn_model(num_classes=len(files))

# Train the model
print("Training the Convolutional Neural Network...")
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
print("Testing the Convolutional Neural Network...")
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\nConvolutional Neural Network Successfully created!\n")

## ---------- CREATING ARTIFICIAL NEURAL NETWORK ---------- ##

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

files = ['apple.npy', 'banana.npy', 'grapes.npy', 'pineapple.npy']
print("Loading data from ", files, "...")
data, labels = load_data(files)

# Split the data into training and test sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)

# Flatten the images for ANN input
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Convert class vectors to binary class matrices (one-hot encoding)
print("Converting labels into categories...")
y_train_cat = to_categorical(y_train, num_classes=4)
y_test_cat = to_categorical(y_test, num_classes=4)

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

# Create and compile the ANN model
print("Creating the Artificial Neural Network...")
ann_model = create_ann_model((x_train_flat.shape[1],), len(files))

# Train the model
print("Training the Artificial Neural Network...")
ann_model.fit(x_train_flat, y_train_cat, batch_size=32, epochs=10, validation_data=(x_test_flat, y_test_cat))

# Evaluate the model
print("Testing the Artificial Neural Network...")
score = ann_model.evaluate(x_test_flat, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\nArtificial Neural Network Successfully created!\n")

## ---------- CREATING RECURRENT NEURAL NETWORK ---------- ##

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import numpy as np
from sklearn.model_selection import train_test_split

files = ['apple.npy', 'banana.npy', 'grapes.npy', 'pineapple.npy']
print("Loading data from ", files, "...")
data, labels = load_data(files)

# Split the data into training and test sets
print("Splitting data into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.05)

# Reshape data for RNN

x_train_rnn = x_train.reshape(x_train.shape[0], x_train.shape[1], -1)  # Reshaping to (samples, time steps, features)
x_test_rnn = x_test.reshape(x_test.shape[0], x_test.shape[1], -1)

# One-hot encode labels
print("Converting labels into categories...")
y_train_cat = to_categorical(y_train, num_classes=len(files))
y_test_cat = to_categorical(y_test, num_classes=len(files))

# Define RNN model
def create_rnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and compile the RNN model
print("Creating the Recurrent Neural Network...")
rnn_model = create_rnn_model(x_train_rnn.shape[1:], len(files))

# Train the model
print("Training the Recurrent Neural Network...")
rnn_model.fit(x_train_rnn, y_train_cat, batch_size=32, epochs=10, validation_data=(x_test_rnn, y_test_cat))

# Evaluate the model
print("Testing the Recurrent Neural Network...")
score = rnn_model.evaluate(x_test_rnn, y_test_cat, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print("\nRecurrent Neural Network Successfully created!\n")

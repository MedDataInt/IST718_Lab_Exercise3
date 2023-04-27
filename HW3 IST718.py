#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Homework3 by Jie Wang 

import tensorflow as tf 
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np 
import matplotlib.pyplot as plt 

print (tf.__version__) 


# In[4]:


# loading the fashion MNIST data 
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# returns 4 numpy arrays: 2 training sets and 2 test sets
# images: 28x28 arrays, pixel values: 0 to 255
# labels: array of integers: 0 to 9 => class of clothings
# Training set: 60,000 images, Testing set: 10,000 images

# class names are not included, need to create them to plot the images  
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[5]:


# exploring and visualiztion the data 
print("train_images:", train_images.shape)
print("test_images:", test_images.shape)


# In[17]:




# Visualize the first 10 images from the training dataset
fig, ax = plt.subplots(2, 5, figsize=(10, 5), gridspec_kw={'height_ratios': [2, 2]}) # to create a 2X5 grid of subplot

# loop through the first 10 images in the training dataset
for i in range(10):  
    row, col = i // 5, i % 5
    ax[row, col].imshow(train_images[i])
    ax[row, col].set_title(class_names[train_labels[i]])
    ax[row, col].axis('off')
plt.tight_layout()
plt.show()


# In[18]:


# Normalizing the data 
# scale the values to a range of 0 to 1 of both data sets.
train_images = train_images / 255.0
test_images = test_images / 255.0


# In[19]:


##########
###
# Training the first NN model
# step1: build the architecuture
# step2: Compile the model
# Step3: Train the model
# Step4: Evaluate the model


# In[21]:


# Step 1- build the architecture 
# model a simple 3 layer neural network using the Keras API of TensorFlow.
model_3 = keras.Sequential([
    keras.layers.Flatten(input_shape = (28,28)), 
    # this layer is used to flatten the input image of shape (28,28) into a 1D array of size '784'
    # this layer does not have any trainable parameters
    keras.layers.Dense(128, activation = tf.nn.relu),
    # this layer is a fully connected layer with 128 units and the Relu activation function
    # this layer has '784*128 + 128 = 100490' trainable parameters, where 784 is the input size and 128 is the output size
    keras.layers.Dense(10, activation = tf.nn.softmax)
    # this layer is another fully connected layer with 10 units and softmax activation function
    # the softmax function is used to convert the output of network to a probability distribution over the 10 class.
    # This layer has 128*10+10=1290 trainable parameters, where 128 is the input size, and 10 is the output size.
])

model_3.summary()


# In[ ]:


# step2 - compile the model 
# compiling the model configures its learning process, including the optimizer, loss function, and metrics
# once the model is compiled, we can train it on the MNIST dataset using the model_3.fit() method
model_3.compile(optimizer = 'adam',  # an adaptive learning rate optimization algorithm
               loss = 'sparse_categorical_crossentropy', # common loss function used for multi-class classification problems
               metrics = ['accuracy']) # trach the accuracy of the model during training and evaluation


# In[22]:


# step3- train the model, by fitting it to the training data 
# 5 epochs, and split the train set into 80/20 for validation 
#epochs=5: the number of times the entire training dataset will be processed during training
# Setting epochs = 5 means that the model will perform five complete iterations over the entire training dataset, updating its parameters after each iteration.
# The number of epochs to use during training is a hyperparameter that can be tuned to optimize the performance of the model on the task at hand.
model_3.fit(train_images, train_labels, epochs = 5, validation_split =0.2) 


# In[23]:


#Step 4 - Evaluate the model
test_loss, test_acc = model_3.evaluate(test_images, test_labels)
print("Model - 3 layers - test loss:", test_loss * 100)
print("Model - 3 layers - test accuracy:", test_acc * 100)


# In[ ]:


#################################################


# In[25]:


# Model a simple 6-layer neural network 
model_6 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#model_6.summary() 
model_6.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[26]:


#Train the NN-6 with 5 epochs 
model_6.fit(train_images, train_labels, epochs=5, validation_split=0.2)

#Evaluate the model with test datasets
test_loss, test_acc = model_6.evaluate(test_images, test_labels)
print("Model - 6 layers - test loss:", test_loss * 100)
print("Model - 6 layers - test accuracy:", test_acc * 100)


# In[ ]:


#########################


# In[27]:


# Model a simple 12-layer neural network 
model_12 = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
#model_12.summary() 
model_12.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[28]:


#Train the NN-12 with 5 epochs 
model_12.fit(train_images, train_labels, epochs=5, validation_split=0.2)

#Evaluate the model
test_loss, test_acc = model_12.evaluate(test_images, test_labels)
print("Model - 12 layers - test loss:", test_loss * 100)
print("Model - 12 layers - test accuracy:", test_acc * 100)


# In[30]:


### We focus on NN with 3 layers 
mode_3_5epochs = model_3.fit(train_images, train_labels, epochs = 5, validation_split =0.2) 


# In[33]:


# make predictions with the model-3
# confidence of the model that the image corresponds to the label 
predictions = model_3.predict(test_images)
predictions.shape 
#(10000, 10), represents the predicted probabilities of each of the 10 classes(labels) for each of 10000 test image
predictions[0] # returns the predicted probabilities for the first test image, represented as an array of length 10


# In[34]:


np.argmax(predictions[0])


# In[35]:


class_names[9]


# In[36]:


#Ankle boot has the highest confidence value 
test_labels[0]


# In[37]:


# plot image in a grid
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
# plot the value array    
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot= plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0,1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# In[38]:


# look at 0th image, predictions, prediction array
i=0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions, test_labels)


# In[39]:


# Plot the first 15 test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
# plt.title("Predictions of the first 15 images, with NN-3")
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)


# In[1]:


## The end of NN model.


# In[2]:


# Try NB  models
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.naive_bayes import GaussianNB


# In[20]:


# Try linear classification  models
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[21]:


# Train Linear Classification classifier
sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000, tol=1e-3, random_state=42)
sgd.fit(x_train, y_train)

# Predict labels for test set
y_pred = sgd.predict(x_test)

# Calculate loss, accuracy, and confusion matrix
loss = np.mean(y_pred != y_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Print results
print("Loss:", loss)
print("Accuracy:", accuracy)
print("Confusion matrix:\n", cm)


# In[31]:


## Nb model
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Define class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load Fashion MNIST dataset
X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)

# Convert pixel values to integers between 0 and 255
X = X.astype('int')

# Split dataset into training and test sets
train_size = 60000
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Train the Naive Bayes model
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# Predict labels for test set
y_pred = gnb.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))

# Plot confusion matrix with class names
fig, ax = plt.subplots(figsize=(10,10))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=class_names,
       yticklabels=class_names,
       xlabel='Predicted label',
       ylabel='True label')
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
# fmt = '.2f' if normalize else 'd'
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
fig.tight_layout()
plt.show()

# Print accuracy
print("Accuracy:", accuracy)


# In[ ]:





## Introduction

Tensorflow is a machine learning library used by researchers and, also, for production. In this guide we’ll explore how to perform simple image classification in Tensorflow using Keras backend.
Image Classification is the task of assigning a single label to an input image from a predefined set of labels, otherwise called labels or categories. After reading this guide you will have a clear understanding of how image classification works. Once Image is classified you can draw the bounding box over the image. This is called Image localization. Image classification is the basis for the self-driving car, creating Generative networks which can help industries in 
decision-making tasks. 



This task can be divided into the following subtasks.
 
 1.  Data Preparation 
 
      A. Load data      
      B.Exploratory data analysis      
      C.Normalization       
      D.Reshaping      
      E. Label encoding      
      F. Split training and validation set

 1.  Building a CNN Model
 2.  
      A. Constructing a sequential CNN model            
      B. Setting hyper parameters      
      C. Setting the optimizer     
      D. Compiling the model      
      E. Fitting the model 
 
 1. Evaluating the Model 
         
 1. Predicting Validation Data

	A. Predicting the Labels for Validation Data
	
	B.  Creating a confusion matrix using predicted and actual labels

1. Predicting Test  Data

 1. Save Predictions
 1. Conclusion
 1. Further Reading
    
##  Environment Setup
 
Google Colab is used for this demo [Google Colab](https://colab.research.google.com/). It provides free GPU and TPU to perform ML/AI tasks. The entire code of this guide can be found in [mnist](https://colab.research.google.com/drive/1nFyGN38mR_y5a5hxsavWw3UelWZhZ1Ka#scrollTo=e0encx4URXIm).  First, you need to grab your New API Token from a Kaggle account. Then, upload the API token file as a kaggle.json in Colab using the following code:
 
```python

from google.colab import files
files.upload()
```

The next step is to mount a Google drive and change it to the desired directory in Google drive. 
 
 ```python
 from google.colab import drive
import os

drive.mount("/content/gdrive")
os.chdir("/content/gdrive/My Drive/<path/of/google drive folder/>")  #change dir

 ```
 
Next, install the Kaggle python library through pip installation. Then, create a directory named Kaggle, copy the API token, and set permissions.

```python
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json  # set permission
```

Then, download MNIST Digit Recognizer data using the below command:

 ```python
 !kaggle competitions download -c digit-recognizer
 ```
 
If the download is successful, three files will be found in the present working directory named “train.csv”, “test.csv”, and “sample_submission.csv”. This data can be downloaded directly from the [Kaggle website](https://www.kaggle.com/c/digit-recognizer/data) and used for training in different environment setup.
 
## About the Data
 
MNIST is a handwritten digit image database, and each image is 28x28 = 784 pixels for each image. The given Digit Recognizer Data has 42000 training images and 28000 test images. Data is represented in CSV format, in which the first column is the label and the remaining 784 columns represent pixel values. Each row represents individual images. The test data contains 784 columns. The task is to predict labels for the 28000 test images; labels are digits 0-9.

 
### 1.  Data Prepration 

#### A. Load Data
   
Read the image data stored in CSV format. The pandas read_csv() function is used to read the CSV file.

  ``` python
  train = pd.read_csv("train.csv")
  test=pd.read_csv("test.csv")
  ```
Then, prepare the data for training by dropping the label column. The training data contains only pixel values.  

  ```python
  X_train = train.drop(["label"],axis = 1)
  Y_train = train["label"]
  X_test=test
  ```

#### B. Exploratory Data Analysis

After reading data check to verify the quality of the data, can you find how the 10 classes in the training images are distributed? Can you find how many missing values are present? The below code counts how many samples are present for each class.    

  ``` python
  g = sns.countplot(Y_train)
  Y_train.value_counts() 
  ```

  ![alt text](https://i.imgur.com/DUsdhUM.png)

Next, we will calculate the number of null values in the training and test data. This will tell us if there are any corrupted images in the data. In this case, there are no null values, so  the data quality is good.


  ``` python
  X_train.isnull().any().describe()
  ```
  ```python
  X_test.isnull().any().describe()
  ```


#### C. Normalization

This is a grayscale image with possible pixel intensity values from 0-255. To make the pixel intensity values within the range 0-1, we’ll divide the intensity values of all pixels by 255. The motivation is to achieve consistency in the range of values being handled to avoid mental distraction or fatigue.

   ``` python
  X_train = X_train/255.0
  X_test = X_test/255.0
  ```
#### D. Reshaping

The Conv2D layers in Keras are designed to work with three-dimensions per image. They have 4D inputs and outputs. The input arguments are the number of samples, width, height, and the number of features or channels. Syntax: reshape (nb_samples,  width, height,nb_features)
  
  ``` python
  X_train = X_train.values.reshape(len(X_train), 28, 28,1)
  X_test = X_test.values.reshape(len(X_test), 28, 28,1) 
  ```
#### E. Label Encoding

When encoding labels, convert labels into one hot encoding. 

![alt text](https://i.imgur.com/wKtY1Og.png)

The Keras function “to_categorical()” takes labels[0-9] as the input and converts it to a [one-hot encoding](https://en.wikipedia.org/wiki/One-hot) of integer encoded values.

  ``` python
  from keras.utils.np_utils import to_categorical

  Y_train = to_categorical(Y_train, num_classes = 10)
   ```
#### F. Split Training and Validation Set

Training data is split into the training and validation set. Validation data is created to evaluate the performance of the model before applying it into actual data. The below code randomly moves 10% of the training data into validation data. We set a random seed =3 to initialize a random generator to randomly pick the validation data.

  ```python
  from sklearn.model_selection import train_test_split
  # Set the random seed
  random_seed = 3
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
  ```

## 2.  Building a CNN Model
 
### A. Constructing Sequential CNN Model

#### i). Convolution Operation

Convolution is done to extract features from input images. In the figure below, the image size is 5x5 and kernel size is 3x3. The kernel is slid over the image to extract the feature map. The feature map that is extracted is spatially correlated.

![alt_text](https://i.imgur.com/NcyYyaJ.gif)

#### ii)Batch Normalization

The batch normalization is used to bring the values in hidden layers into the same scale as everything else. To classify oranges from lemons, each batch sees a different set of values and their activation values will be different. Batch Normalization reduces the dependency between each batch by bringing all of the values into the same scale. 

![alt_text](https://i.imgur.com/oimZurP.png)

#### iii)Max Pooling

Max Pooling extracts important features obtained from convolution. Max pooling is done after a few convolutions. In the code below, 2x2 max pooling is used. It finds the maximum value in the 2x2 and returns the highest value. It also reduces the number of parameters in the network by reducing the size of the feature map.
![alt_text](https://i.imgur.com/PIF7Fmn.png)

#### iv) Global Average Pooling

For the 11x11x10 incoming tensor feature maps take the average of each 11x11 matrix slice which gives a 10-dimensional vector. This can feed into the fully-connected layers which are a single-dimension vector representing 10 classes.

#### v) ReLu Activation

The ReLu Activation function is used to carry forward all of the positive values to the next layer and makes sure negative values are dropped down. Any value less than zero is negative; a value of zero or greater is taken as a positive value. 

![alt_text](https://i.imgur.com/PMXrkfO.png)

```python

model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(28,28,1))) # 26
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu')) # 24
model.add(BatchNormalization())

model.add(Conv2D(50, (3, 3), activation='relu')) # 22
model.add(BatchNormalization())

model.add(Conv2D(52, (3, 3), activation='relu')) # 20
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu')) # 18
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu')) # 16
model.add(BatchNormalization())

model.add(Conv2D(27, (3, 3), activation='relu')) # 14
model.add(BatchNormalization())

model.add(Conv2D(15, (3, 3), activation='relu')) # 12
model.add(BatchNormalization())

model.add(Conv2D(10, (3, 3), activation='relu')) # 9
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())


#model.add(Flatten())
model.add(Activation('softmax'))

#model.add(Flatten())
model.add(Activation('softmax'))
```
### B. Setting Hyperparameters

A hyperparameter is a parameter whose value is set before the learning process. The hyperparameters present in CNN are:

* Learning Rate
* Number of Epochs - Number of times the whole training image is seen
* Batch Size - Number of images to be read at a time for extracting feature maps

### C. Setting the Optimizer
The optimizer is used to update weight and model parameters to minimize the loss function. Adam stands for Adaptive Moment Estimation and it’s chosen because of its fast convergence. 

### D. Compiling the Model
When compiling the three parameters  loss, the optimizer and metrics are required. 

- categorical_crossentropy is a loss function for categorical variables
- Use the Adam Optimizer to control the learning rate
- The metric 'accuracy' is used to measure the performance of the model

``` python
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
```

### E. Fitting the Model 

Apply the model in the training and validation set.

``` python
%%time
history = model.fit(X_train, Y_train, epochs=40,verbose=1,validation_data = (X_val,Y_val),batch_size=batch_size)
```

## 3. Evaluating the Model

Now find the validation loss and validation accuracy of the model. 

``` python
val_loss,val_acc = model.evaluate(X_val, Y_val, verbose=0)
print("Validation Accuracy:",val_acc)       
```    

The validation accuracy is 99.19 %. 


## 4. Predicting  Validation  Data

### A. Predicting the Labels for Validation Data

``` python
# Predict the values from the validation dataset
Y_pred = model.predict(X_val)
```

### B. Creating a Confusion Matrix Using Predicted and Actual Labels

``` python
import itertools 

# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(Y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10))  
```     

## 5. Predicting Test Data

This time predict classes for unseen images. These images are not used in training or validation. 

```python
# predict results
results = model.predict(X_test)
# select the index with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
```
### 6. Save Predictions

The predicted labels are stored in a CSV file using the pandas to_csv function.

```python
submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
submit.to_csv("cnn_mnist_predictions.csv",index=False)
```



## Conclusion

In this guide, we have learned how to load images from Keras datasets, how to preprocess images, how to train images, validate the model performance and how to predict classes for unseen images. You can change the model hyperparameters and train it again, and see if the performance can be further increased.

## Further Reading

To know more about image classification techniques you can read about the different types of convolution architecture. Densenet and its variants are widely used for Image Classification, Image Segmentation, Image Localization tasks. 



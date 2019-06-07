### Getting Started With Tensorflow

Tensorflow is a machine learning library used by researchers and also for production. In this guide how to perform simple image classification in tensorflow using keras backend is explained.
 Image Classification is the task of assigning single label to  input image from predefined set of labels otherwise called labels or categories. This task can be divided into following sub tasks.
 
 1.  Data Prepration 
      1. Load Data
      1. Exploratory data analysis
      1. Normalization
      1. Reshaping
      1. Label encoding
      1. Split training and validation set

 1.  Building CNN Model
     1. Constructing sequential CNN model
     1. Set hyperparameters
     1. Set optimizer
     1. Compiling the model
     1. Fit the Model 
 
 1. Evaluate the model
     1. Find training and validation accuracy
     1. Creating confusion matrix using predicted and actual labels

 1. Image Prediction
     1. Predict label for given image
    
 ####  Environment Setup
 
Google Colab is used to do this demo. <url>https://colab.research.google.com/<url>. Google Colab provides free GPU and TPU to perform ML/AI tasks. First you need to grab your New API Token fromm Kaggle account. Then uplaod api token file kaggle.json in colab using 
 
```python

from google.colab import files
files.upload()
```



To load data create
 
 Next step is to mount a google drive and change to desired directory in google drive. 
 
 ```python
 from google.colab import drive
import os

drive.mount("/content/gdrive")
os.chdir("/content/gdrive/My Drive/<path/of/google drive folder/>")  #change dir

 ```
 
 Netx install kaggle python library through pip installation then create a directory named kaggle and copy api token and set permissions 

```python
!pip install -q kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!ls ~/.kaggle
!chmod 600 /root/.kaggle/kaggle.json  # set permission
```

Then download MNIST Digit Recognizer data using below command
 ```python
 !kaggle competitions download -c digit-recognizer
 ```
 
 If download is successful three files found in present working directory namely train.csv, test.csv and sample_submission.csv. This data can be dowloaded directely from Kaggle website <url>https://www.kaggle.com/c/digit-recognizer/data and used when training in different environment setup.
 
 ### About the Data
 
 MNIST is a hand written digits images and each image is of size 28x28 = 784 pixels for each image. Given Digit Recognizer data has  42000 training images and 28000 test images. Data is represented  in csv format in which first column is label and remaining 784 columns represent pixel value. Each row represent individual images. Test data contain 784 columns. The task is to predict labels for the 28000 test images. Labels are digits 0-9.

 
1.  Data Prepration 
      1. Load Data
     Read image data stored in csv format. Pandas read_csv() function is used to read csv file.
      ``` python ```
      train = pd.read_csv("train.csv")
      test=pd.read_csv("test.csv")
      ```
  Then prepare data for traininig by dropping the label column. The training data contains only pixel values.  
      
      ```python
      X_train = train.drop(["label"],axis = 1)
      Y_train = train["label"]
      X_test=test
      ```
      
 1. Exploratory data analysis
 
 After reading data check the quality of data.Find how the 10 classes in training images are distributed?, Find how many missing values present?. The below code counts how many samples present for each classes.    
 
``` python
g = sns.countplot(Y_train)
Y_train.value_counts() 
```
      
![alt text](https://github.com/joyjeni/AI/blob/master/session1/img/class_count.png "EDA")

Next we calculate number of null values in train and test data. This will tell is there any corrupted images in data. In this case there is no null values the data quality is good.


``` python
X_train.isnull().any().describe()
```
```python
X_test.isnull().any().describe()
```

 1. Normalization
  
 This is gray scale image with possible pixel intensity values from 0-255. To make the pixel intensity values within range 0-1 divide all pixel intensity values by 255. The motivation is to achieve consistency in range of values handled to avoid mental distraction or fatigue
 
 ``` python
X_train = X_train/255.0
X_test = X_test/255.0
```
1. Reshaping

The Conv2D layers in Keras is designed to work with 3 dimensions per image. The have 4D inputs and outputs. The input arguments are number of samples, width,height and number of features or channels. Syntax: reshape (nb_samples,  width, height,nb_features)

  ``` python
X_train = X_train.values.reshape(len(X_train), 28, 28,1)
X_test = X_test.values.reshape(len(X_test), 28, 28,1)
```
1. Label encoding

In label encoding convert labels into one hot encoding. 

![alt text](https://github.com/joyjeni/AI/blob/master/session1/img/onehot_cropped.png "onehot")

Keras function to_categorical() takes labels[0-9] as input and convert to one hot encoding of integer encoded values.

``` python
from keras.utils.np_utils import to_categorical

Y_train = to_categorical(Y_train, num_classes = 10)
   ```
 1. Split training and validation set
 
 Training data is splitted into train and validation set. Validation data is created to evaluate the performance of model before applying it into actual data. Below code randomly moves 10% of training data into validation data. We set random seed =3 to initialise random generator to randomly pick the validation data.
  
  ```python
from sklearn.model_selection import train_test_split
# Set the random seed
random_seed = 3
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.1, random_state=random_seed)
```

 1.  Building CNN Model
     1. Constructing sequential CNN model
     
     * Receptive Field
     
     * Convolution 
     * Batch Normalization
     * Max Pooling
     * Global Average Pooling
     * Activation
     
``` python 
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) # 26
model.add(BatchNormalization())

model.add(Conv2D(16, (3, 3), activation='relu')) # 24
model.add(BatchNormalization())

model.add(Conv2D(21, (3, 3), activation='relu')) # 22
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2))) #11

model.add(Conv2D(18, (3, 3), activation='relu')) # 9
model.add(BatchNormalization())



model.add(Conv2D(27, (3,3), activation='relu'))#7
model.add(BatchNormalization())

model.add(Conv2D(15, (3,3), activation='relu'))#5
model.add(BatchNormalization())

model.add(Conv2D(10, (3,3), activation='relu'))#3
model.add(BatchNormalization())


model.add(Conv2D(10, 1, activation='relu'))#1
model.add(BatchNormalization())


model.add(GlobalAveragePooling2D())


#model.add(Flatten())
model.add(Activation('softmax'))
```

1. Set hyperparameters
     ``` python 
      
      ```
     1. Set optimizer
     ``` python
      
      ```
     1. Compiling the model
     
 ``` python
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
      ```
     1. Fit the Model 
     
``` python
%%time
history = model.fit(X_train, Y_train, epochs=40,verbose=1,validation_data = (X_val,Y_val),batch_size=batch_size)
      ```
 
 1. Evaluate the model
 
     1. Find training and validation accuracy
     ``` python
      val_loss,val_acc = model.evaluate(X_val, Y_val, verbose=0)
      print("Validation Accuracy:",val_acc)       
      ```    
      
 The validation accuracy is 99.11 %. 
     
      

 1. Image Prediction
     1. Predict label for given image
     ``` python ```
      # Predict the values from the validation dataset
Y_pred = model.predict(X_val)
      ```
      
     1. Creating confusion matrix using predicted and actual labels
     
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
      
      1. Visualize Top 25 Errors
      
   ```python
   ```
      ![top25](https://github.com/joyjeni/AI/blob/master/session1/img/top25_errors.jpg top25)
      
    1. Predict Test Data
    ```python
    # predict results
results = model.predict(X_test)
# select the indix with the maximum probability
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
```


```python

submit = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submit.to_csv("cnn_mnist_predictions.csv",index=False)
```





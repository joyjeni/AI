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
     1. Visualize Top 10 errors
 
 1. Further Optimization
 
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
 
 MNIST is a hand written digits images and each image is of size 28x28 = 784 pixels for each image. Given Digit Recognizer data has  42000 training images and 28000 test images. Data is represented  in csv format in which first column is label and remaining 784 columns represent pixel value. Each row represent individual images. Test data contain 784 columns. The task is to predict labels for the 28000 test images.  

 

# Emotion_recognition_CNN
###
### Prerequisites
Install these prerequisites before proceeding-
```
 pip3 install tensorflow
 pip3 install keras
 pip3 install numpy
 pip3 install sklearn
 pip3 install pandas
 pip3 install opencv-python
```
###
### Requirements

- Python 3.3+ or Python 2.7
- macOS or Linux 

## Build from scratch

Clone this repository using-
```
git clone https://github.com/abhijeetghubade/Emotion_recognition_CNN.git
```

### Download the Dataset
Download and extract the [Face Emotion Recognition (FER)](https://drive.google.com/open?id=12LH4RkgIrZ7Vbk8VQU22XQ-4AlS2fwfE) dataset.

### Preprocessing and Training

Run the [`preprocessing.py`](https://github.com/abhijeetghubade/Facial_Emotion_recognition_CNN/blob/master/preprocessing.py) file, which would generate `features.npy` and `labels.npy` files for you.
Run the [`fertrain.py`](https://github.com/abhijeetghubade/Facial_Emotion_recognition_CNN/blob/master/fertrain.py) file,  this would take sometime depending on your processor and gpu. Took around 1 hour for with an Intel Core i7-7700K 4.20GHz processor and an Nvidia GeForce GTX 1060 6GB gpu, with tensorflow running on gpu support. If you don't have required hardware, you can use "Google Colab" for Training. It is free of cost and easy to use (I used it).

This would create `modelXtest.npy`, `modelytest,npy`, `fer.json` and `fer.h5` file for you.

## Running the tests (Optional)

You can test the accuracy of trained classifier using `modXtest.npy` and `modytest.npy` by running [`fertest.py`](https://github.com/abhijeetghubade/Facial_Emotion_recognition_CNN/blob/master/fertest.py) file. This would give youy the accuracy in % of the recently trained classifier.
This Model -  66.369% accuracy

You can do the same on your custom test image and running the [`fertestcustom.py`](https://github.com/abhijeetghubade/Facial_Emotion_recognition_CNN/blob/master/fertestcustom.py) file. To make things more fun, I tested the model on faces of the cast from a popular TV Series F.R.I.E.N.D.S and results were pretty good!

![](https://github.com/abhijeetghubade/Emotion_recognition_CNN/blob/master/combined%20results.png)

## Future Usage:
I used the Convolutional Neural Network (CNN) approach to get features from images. In future work, we can look into any other Deep Learning approach to do the same. We can also make some improvement, and make it work in real time.
Although this model performs rather well, to improve its log loss we need to train our model by adding more images.

### Trying to Make it easy to use
I am trying deploy this model on a website, using a lightweight WEGB (Web Server Gateway Interface) web application framework [`Django`](https://www.palletsprojects.com/p/flask/)



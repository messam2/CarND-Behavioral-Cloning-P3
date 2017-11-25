**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py contains the model that has been used, generator for the training and validation data and procedures for traing the model.
* model.h5 the last successful model.
* video.mp4 video of the car drive autonomously in track 1.
* drive.py for driving the car in autonomous mode
* README.md summarizing the results.

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model uses Nvidia CNN architecture with dropout layr to overcome overfitting.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated by the sample data provided by udabity. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, left and rigth side cameras with correction value to recover from left and right sides of the road.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to compare diffrent models like LeNet, nvidia. 

My first step was to use a convolution neural network model similar to the LeNet. This model was fine but not with sharp curvs, so I introduced some complexity by using Nvidia model.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that I added dropout layer.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, I increased the nuber of epoches used.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes.

|Layer (type) | Output Shape | Param # |
|-------------|--------------|---------|
cropping2d1 (Cropping2D)|(None, 100, 320, 3)|0
lambda1 (Lambda)|(None, 100, 320, 3)|0
conv2d1 (Conv2D)|(None, 48, 158, 24)|1824
conv2d2 (Conv2D)|(None, 22, 77, 36)|21636
conv2d3 (Conv2D)|(None, 9, 37, 48)|43248
conv2d4 (Conv2D)|(None, 7, 35, 64)|27712
conv2d5 (Conv2D)|(None, 5, 33, 64)|36928
flatten1 (Flatten)|(None, 10560)|0
dropout1 (Dropout)|(None, 10560)|0
dense1 (Dense)|(None, 100)|1056100
dense2 (Dense)|(None, 50)|5050 
dense3 (Dense)|(None, 10)|510 
dense4 (Dense)|(None, 1)|11  

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I used the data set provided by udacity.

To augment the data set, I used the side cameras images with adding correction value.

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 30 and 24 for the batch size. I used an adam optimizer so that manually training the learning rate wasn't necessary.

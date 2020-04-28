# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/cnn-architecture.png "Model Visualization"
[image2]: ./examples/standard_mid.jpg "Mid Camera"
[image3]: ./examples/standard_left.jpg "Left Camera"
[image4]: ./examples/standard_right.jpg "Right Camera"
[image5]: ./examples/jungle_mid.jpg "Mid Camera Jungle"



---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) network architecture from Nvidia
The network consists of a normalization layer, followed by 5 convolutional layers, followed by 4 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

I included a Dropout layer with a probability of 40% (probability of setting outputs from last hidden layer to zero)
 before the first Dense layer.

The model was trained and validated on different data sets to ensure that the model was not overfitting.
Additionally to the provided data set, I recorded two rounds on the standard track (one clockwise and one counter-clockwise)
and one round on the jungle track. I used these recorded data for my model training.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track (see video.mp4).

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used center lane driving in the two different track scenarios.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Switching from a LeNet architecture to Nvidias Network architecture mentioned before and also cropping out the top of the image
improved the driving behaviour a lot and was much more fluently.

After the first steps of training with the provided training data and the Nvidia Architecture I run the simulator to see how well the car was driving around track one. 
There were a few spots where the vehicle fell off the track, especially at sharp turn e.g. near the lake in the standard track it always drove towards it.

To improve the driving behavior in these cases, I included the side cameras and added an steering offset of +-0.2 for side camera images. 
I tried to increase the steering offset to 0.3, but then the car did not drive in the center of the lane so I reverted back to 0.2.
Also I implemented the recorded data of a counter-clockwise lap and a lap on the jungle track for network training.
After these two implementations the model could also take the sharp turns near the lake.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then included the side camera images and implemented the steering offset in network training, here you can see examples of the left side and right side camera:

![alt text][image3]
![alt text][image4]

After it I recorded and implemented one lap on the second track, here is an example image of the jungle track of the training data, 
which helps generalize the model and avoids overfitting to the one specific track :
![alt text][image5]

You can see my result in the attached video.mp4 where the car drives itself around the track one full lap.

# **Behavioral Cloning** 

---

** Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./Track1_center.png "Track1 center lane driving"
[image2]: ./Track1_center_cropped.png "Cropped image"
[image3]: ./Track2_recovery_centercamera.jpg "Track2 recovery image - center camera"
[image4]: ./Track2_recovery_rightcamera.jpg "Track2 recovery image - right camera"
[image6]: ./Track2_recovery_center_flipped.png "Track2 flipped image"
[image7]: ./histogram_trainingdata.png "Histogram of training data"
[image8]: ./error_curve.png "Train/Validation loss with and without maxpooling"

** Submission **

#### 1. The following files are included and can be used to run the simulator in autonomous mode

* model.py containing the script to create and train the model
* data_processing.ipynb (and exported html file) containing the script to combine, augment, visualize training data.
* drive.py for driving the car in autonomous mode
* model-final.h5 containing a trained convolution neural network 
* final_report.md summarizing the results
* track1.mp4, and track2.mp4 video files showing the vehicle driving autonomously in both tracks.
* Other images used for the writeup

#### 2. Running the code:
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around
tracks 1 and 2, by executing 'python drive.py mode-final.h5'.

#### 3. Code description:

The model.py file contains the code for training and saving the convolution neural network. The file
shows the pipeline I used for training and validating the model. 

The file 'data_processing.ipynb' contains code for training data augmentation, mixing, histogtam
plots etc.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the NVIDIA network and tune its parameters as well as collect appropriate
training data so that the model drives the car autonomously on both tracks. The NVIDIA architecture
seemed a good starting point as it has been known to be used in practise.

The network had 5 convolutional layers and three fully connected layers, with dropouts, and preprocessing.
The filter sizes for the first three layers are 5x5 and the next two layers are 3x3. The depths are
between 24 and 64 (model.py lines 89-93).

The model includes RELU layers to introduce nonlinearity (code line 89), and the data is normalized in 
using a Keras lambda layer (code line 87). The image is also cropped (50 pixels from top, and 20
from bottom; line 88). Note that RELU activation is applied only for the convolutional layers and not the fully
connected layers.

The strides for the convolutional layers are 2x2 (lines 89-93). Initially, I considered a 1x1 stride followed
by 2x2 maxpooling layer after each convolutional layer. But, I figured that this model has higher
training loss. The plot below shows training loss with and without maxpooling on the training data only from track 1.
As we can see, maxpooling layers contributes to higher training loss.

![Error curves with and without maxpooling][image8]

Therefore, I decided to avoid maxpooling. Morevover, I also noticed that the validation loss was 
typically higher than the training loss, so the model seemed to overfit. Therefore, I decided to
use dropouts to avoid overfitting. I noticed few things, while using dropouts.
Firstly, adding dropouts near the convolutional layers did not help. Rather it was causing higher training
loss. So, I decided to add dropouts only to the fully connected layers. Secondly, dropouts did not
provide considerable reduction in validation loss for most choices of dropout rate. But, it did help a bit
if the dropout rate is low. So, I added dropouts added near the fully connected layers (line 94 and 97).

Finally, another layer that I experimented with is the size of the cropping layer. I found that
cropping 50 pixels from top, and 20 from bottom seemed to give good driving behaviour on both
tracks. If I chose higher 50, the model was slow in predicting steering angles (as expected).
The size of fully connected layers are 100, 50, and 1.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the
following layers and layer sizes:
0. Layer 0 - Lambda layer for normalization, and a cropping layer (line 86, 87)
1. Layer 1 - convolutional layer with 5x5 kernel, and 24 features
2. Layer 2 - convolutional layer with 5x5 kernel, and 36 features
3. Layer 3 - convolutional layer with 5x5 kernel, and 48 features
4. Layer 4 - convolutional layer with 3x3 kernel, and 64 features
5. Layer 5 - convolutional layer with 3x3 kernel, and 64 features
6. Layer 6, 7, 8 - Fully connected layers with dropouts (lines 83 to 88). 

The final model was trained and validated on data sets from both tracks (as dscussed in Section 3), to ensure
that the model was not overfitting. The model was tested by running it through the simulator and
ensuring that the vehicle could stay on the track.

#### 3. Creation of the Training Set

At first recorded two laps of center lane driving on both tracks. An example of center lane 
driving on track 1 is shown below:

![Track 1 center lane driving][image1]

In the next step, I tried a recovety lap on both tracks, where I allowed the vehicle to 
go towards the edge of the lane, and then steered it back. Finally, I collected additional
data by driving around sharp curves on both tracks. The model initially had much difficulty driving
around the curves in track 2, but with additional training data around those curves, the model
was finally able to drive the car autonomously on both tracks.

The figures below show recovery image for Track 2, from center and right cameras.
![Center camera][image3]

![Right camera][image4]

After collecting training data, I also augmented it by flipping images and also choosing
left or right camera with probability of 0.5. An example of flipped image and cropped image is
shown below. The flipped image corresponds to the track 2 right camera recovery image above.
And the cropped image corresponds to the Track 1 center lane driving shown at the beginning of
Secion 3.

![Track 2 center camera image flipped][image6]

![Track 1 image cropped][image2]

Finally, I set a threshold for the number of images with steering angle close to 0. I did this
to allow the model the model to train effectively for all steering angles. The histogram of
sample size for different steering angles is below:

![Histogram of training data][image7]

In the end, I had about 21000 data points, which I split into to training/validation set,
with 80% in training set.

The python notebook (data_processing.ipynb) contains code for training data augmentation and 
mixing.

#### 4. Training Process
I used a batch size of 128. Loss function was 'mse' and the optimizer was adam optimizer (with default learning
rate of 0.001). I ran the model
for 12 epochs, as I found that the validation loss seemed to saturate or sometimes even increase,
while the training loss continued to decrease (similar to loss curves in Section 1). I did not want the model
to overfit. So, I stopped with 12 epochs (line 91-92). The training loss and validation
loss for the final model on combined training data was 0.0264 and 0.0614 respectively.

The model was trained on AWS GPU instances and tested by running it through the simulator.
It was able to drive the vehicle autonomously on both tracks.

For training the model, use 'python  model.py  --model_name  model-final.h5  --num_epochs 12'.
For driving the car autonomously, use 'python drive.py model-final.h5'.

The video recording for both tracks are in files track1.mp4, and track2.mp4 respectively.

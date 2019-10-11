# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/dataset_stat.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/image1.jpg "Stop Sign"
[image5]: ./examples/image2.jpg "Speed limit (70km/h)"
[image6]: ./examples/image3.jpg "Priority road"
[image7]: ./examples/image4.jpg "Turn left ahead"
[image8]: ./examples/image5.jpg "Roundabout mandatory"
[image9]: ./examples/lenet.png "LeNet CNN framework"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/ed8808/CarND-Traffic-Sign-Classifier/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The RGB is firstly converted to Grayscale followed by local historgram equalization to increase contrast of training set.
Finally to perform normalization of the image data from 0..255 to -1..1. 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten  | outputs 400 |
| Fully connected		| outputs 120				|
| RELU					|												|
| Fully connected		| outputs 84				|
| RELU					|												|
| Fully connected		| outputs 43				|

![alt text][image9]

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, with 
* batch size = 128
* epochs = 10 with early termination detection if the current accuracy is less than last one by 5%
* learning rate = 0.001
* dropout at fully connected layers with keep_prob = 0.5
* L2 regularization added to all Weights with beta factor set to 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.5%
* validation set accuracy of 95.5%
* test set accuracy of 93.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? 

The original LeNet CNN framework was chosen from LeNet lab session as a ground reference.

* What were some problems with the initial architecture?

The architecture remains unchanged, although the performance is relied heavily on tweaking the parameters such as learning rate, regularization and dropout.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Regularization and dropout are introduced to avoid under fitting or over fitting.

* Which parameters were tuned? How were they adjusted and why?

The learning rate has been found crucially important to the training performance.  Besides the pre-processing of images should filter out less important parameters such as color by using grayscale.  To increase contrast, local histogram equalization can boost up the accuracy as well.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Half of the tensors are dropout at each fully connected layers after the activation function

If a well known architecture was chosen:
* What architecture was chosen?

LeNet-5 was explored in this project.

* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

The validation accuracy is 95.5% while test accuracy is 93.4%, so there is a very slightly over-fitting but not too much to be considered as true over-fitting.  The final test of 5 new images has test accuracy = 100%.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

When the new images were firstly introduced to the network, the test accuracy was literally zero without any clue until the issue had been figured out from poor image quality of new images used.  Therefore the image quality plays a part in CNN.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Speed limit (70km/h)     			|  			Speed limit (70km/h)							|
| Priority road		| Priority road				|
| Turn left ahead	      		| 	Turn left ahead				 				|
| Roundabout mandatory		| Roundabout mandatory     							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the 1st image, the model is almost 100% sure that this is a stop sign (probability of 1), and the image does contain a stop sign. The top five soft max probabilities were

14, 17, 38, 34, 15
9.99998927e-01, 4.38427207e-07, 3.36586879e-07, 2.71290332e-07, 4.42601582e-08

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Stop sign   									| 
| 0     				| No entry 										|
| 0					| Keep Right										|
| 0	      			| Turn Left					 				|
| 0				    | No Vehicle      							|


For the 2nd image, the model is 82% sure that this is a speed limit of 70km/h. The top five soft max probabilities were

4,  0,  1, 40, 37
8.23942721e-01, 1.04461856e-01, 7.10061863e-02, 2.30392776e-04, 2.18037283e-04

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.82         			| Speed limit (70km/h)   									| 
| 0.10     				| Speed limit (20km/h) 										|
| 0.07					| Speed limit (30km/h)											|
| 0	      			| Roundabout mandatory				 				|
| 0				    | Go straight or left     							|

For the 3rd image, the model is almost 100% sure that this is a Priority road
(probability of 1). The top five soft max probabilities were

12, 40, 11, 13,  9
9.99999881e-01, 9.89836337e-08, 2.20364988e-08, 3.06537440e-09, 1.39319989e-09

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| Priority road  									| 
| 0     				| Roundabout mandatory										|
| 0					| Right-of-way at the next intersection											|
| 0	      			| Yield			 				|
| 0				    | No passing      							|

For the 4th image, the model is 92% sure that this is a Turn left ahead. The top five soft max probabilities were

34, 35, 12, 13, 41
9.15628493e-01, 2.57545989e-02, 1.21714631e-02, 1.03315990e-02, 9.84210707e-03

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.92         			| Turn left ahead   									| 
| 0.26     				| Ahead only 										|
| 0.12					| Priority road											|
| 0.10	      			| Yield			 				|
| 0.01				    | End of no passing      							|

For the 5th image, the model is 0.44 sure that this is a Roundabout mandatory.  The top five soft max probabilities were

40, 37, 10, 18, 33
4.35587525e-01, 1.69787884e-01, 1.39661103e-01, 3.72859277e-02, 3.34356166e-02

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.44         			| Roundabout mandatory   									| 
| 0.17     				| Go straight or left										|
| 0.14					| No passing for vehicles over 3.5 metric tons											|
| 0.04	      			| General caution			 				|
| 0.03				    | Turn right ahead      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/multi-scale-windows.png
[image2]: ./output_images/example.jpg
[image3]: ./output_images/window1.png
[image4]: ./output_images/heat1.png
[image5]: ./output_images/label1.png
[image6]: ./output_images/hotwindow1.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

All the code for this project is contained in the IPython notebook located in "./vehicle_detection.ipynb".

### Feature Extraction

#### 1. Extract features from the training images.

I used all techniques learned from the course: color histograms, bin colors, histograms of gradiensts.

I tested with all three features and it proved to improve algorith robustness for identifying color, shape, and both.

#### 2. Explain how you settled on your final choice of HOG parameters.

With all code provided from the course, I tried out combinations of params using itertools.product.

Firstly, I set fixed color space & hog channel as 'RGB' & 'ALL'. Then I tried different combinations of enable/disable certain features. I found out that using three features together would achieve the best result.

Then I tested using different combinations of color and hog channel, and found out the value set of 'YCrCb' and 'ALL' has the best results.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Before training the classifier, I took the advice from the Udacity to deal with the time series data. I created dataset as below to to avoid nealy identical timeseries data being included in bote train and test datasets.

train dataset = random picked non-timeseries data + all timeseries data
test dataset = random picked non-timeseries data

With LinearSVC as classifier, I am able to achieve 99.02% accuracy.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I searched windows for region where car is possible to show up: x=(0, 1280), y=(390, 650).

Then I did mutliple scale for search windows. For example, window with size < 70 will only capture cars very far from the camera, so it will not show up in regions of too left (x_left < 600), too right (xbox_left > 930), and too below (ytop > 420).

I tested with a few overlap values and picked 0.85.

![Multi scale windows][image1]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched windows on four scales: 1.1, 1.5, 1.9, 2.3. Within each window, the image is resized to (64, 64) using opencv, transform image to HLS color image, then extract all channel HOG feature + binned color + color histograms. All features is put into a feature vector, then feed it to SVM classifier to predict if it is a car or not.

The below example shows the output of all windows found for an image.

![Original image][image2]
![Window image][image3]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [video1](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

With all window boxes with region image being classified as a car image, I created heatmap from these boxes. Then I applied a threshold to find hot windows from the heat map, then draw image with hot windows.

To avoid randomly changing hot windows over timeseries images, I saved 5 most recent image frames and compute average for the heat map.

### Here are six frames and their corresponding heatmaps:

[Example1 window image]: ./output_images/window1.png
[Example1 heat image]: ./output_images/heat1.png

[Example2 window image]: ./output_images/window2.png
[Example2 heat image]: ./output_images/heat2.png

[Example3 window image]: ./output_images/window3.png
[Example3 heat image]: ./output_images/heat3.png

[Example4 window image]: ./output_images/window4.png
[Example4 heat image]: ./output_images/heat4.png

[Example5 window image]: ./output_images/window5.png
[Example5 heat image]: ./output_images/heat5.png

[Example6 window image]: ./output_images/window6.png
[Example6 heat image]: ./output_images/heat6.png

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
[Example1 label image]: ./output_images/label1.png
[Example2 label image]: ./output_images/label2.png
[Example3 label image]: ./output_images/label3.png
[Example4 label image]: ./output_images/label4.png
[Example5 label image]: ./output_images/label5.png
[Example6 label image]: ./output_images/label6.png

### Here the resulting bounding boxes are drawn onto the last frame in the series:
[Example1 hot window image]: ./output_images/hotwindow1.png
[Example2 hot window image]: ./output_images/hotwindow2.png
[Example3 hot window image]: ./output_images/hotwindow3.png
[Example4 hot window image]: ./output_images/hotwindow4.png
[Example5 hot window image]: ./output_images/hotwindow5.png [Example6 hot window image]: ./output_images/hotwindow6.png


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I believe my pipeline meets the project requirement that it performs reasonably well as it identify the vehicles most of the time, and it has no false positives!

However, it still has a few issues.

First, there are about two frames that a car is not identified. I doubt the params I chose is "local optimal" and there might be better chooice.

[Missing Example1]: ./output_images/missing1.png
[Missing Example2]: ./output_images/missing2.png

Second, when a car gets out of the image, the window chanegs a lot and this might be fixed if I picked window more carefully.

[Missing Example3]: ./output_images/missing3.png

Third, when a car is far away, like in the beginning of the video, there is car very far away and is unble to be captured. I think this might be fixed if picking a smaller sliding window size like 32.

[Missing Example4]: ./output_images/missing4.png

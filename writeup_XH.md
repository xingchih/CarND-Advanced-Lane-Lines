##Project 4 Advanced Lane Finding Writeup 
###Xingchi He

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/cam_cal_undist_chess.png "cam_cal_undist_chess"
[image2]: ./output_images/cam_cal_undist_car.png "cam_cal_undist_car"
[image3]: ./output_images/color_grad_combined_thresh.png "color_grad_combined_thresh"
[image4]: ./output_images/perspective_src_pnts.png "perspective_src_pnts"
[image5]: ./output_images/perspective_transform_example.png "perspective_transform_example"
[image6]: ./output_images/sliding_window.png "sliding_window"
[image7]: ./output_images/video_output_screen_shot.png "video_output_screen_shot"
[video1]: ./project_video_output.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the code cells 2-4 of the IPython notebook "./CarND_Advanced_Lane_Lines.ipynb".  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `obj_points` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `img_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `obj_points` and `img_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To correct camera distortion, I used the camera matrix and the distortion coefficients obtained from the chessboard camera calibration and applied `cv2.undistort()` to get the undistorted image. Below is an example of the pre and post undistortion images:
![alt text][image2]

####2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (function `color_gradient_threshold()`, code cell 8 and 9 in `./CarND_Advanced_Lane_Lines.ipynb`).  Here's an example of my output for this step. 

![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

For perspective transform, I use `cv2.getPerspectiveTransform()` function that takes as inputs source (`src`) and destination (`dst`) points and generates `M` and `Minv` for the perspective and its inverse transforms. THe example can be found in code cell 9-13. These matrix will be used later to perform the perspective transform with function `cv2.warpPerspective()` in code cell 27. I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I overlayed the `src` points onto two test images of straight lines and verified those points match the lane lines, as shown in the figures below:

![alt text][image4]

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image5]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used the sliding window method and fit my lane lines with a 2nd order polynomial. The function can be found in  The figure below shows the sliding windows (green), the identified pixels for left and right lanes (red and blue), and the fitted 2nd order polynomial (yellow) in a top-down view:

![alt text][image6]

This example can be found in code cell 14-20 and the actual process in the pipeline is in code cell 26, function `polynomial_fit()` in `CarND_Advanced_Lane_Lines.ipynb`.

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `calc_curv()' in code cell 26 of `CarND_Advanced_Lane_Lines.ipynb`. This calculates the radius of curvature in meters.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `draw_lane()`.  Here is an example of my result, as a screen shot from the project video result:

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_output.mp4) from `project_video.mp4`
Video results from `challenge_video.mp4` and `harder_challenge_video.mp4` can be found [here](./challenge_video_output.mp4) and [here](./harder_challenge_video_output.mp4).

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The approach I took consists of the following steps:

* Apply a distortion correction to raw images. The camera calibration calibration matrix and distortion coefficients were obtained through chessboard calibration.
* Use color and gradient thresholding to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit a second order polynomial to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Apply a moving average filter to the lane fitting, the estimation of radius of curvature and the estimation of the car position with respect to the center of the lane
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

These steps worked well for the project video, which has relatively clearly marked lanes, very gentle turns (large radius of curvature), relatively steady lighting, minimal change in the dynamic range (shadow and saturation of the images), and very little noise (no patch in the lane, divider does not hijack the lane estimation). The same algorithm has trouble with the challenge video `challenge_video.mp4`. The other challenge video `harder_challenge_video_output.mp4` has very tight turns, drastic change in exposure, high constrast noise, visual features from background environment that exhibits similar features as lane lines through my algorithm. All these factors made it very difficult for my algorithm to robustly track the lane lines. It seems likely that through extensive experimenting, I might be able to find a set of parameters (color and gradient thresholds, perspective transform, curvature condition, higher order of polynomial fit, and so on) that provide improved results. But it's unlikely this would generalize to other corner cases. It would be interesting to know what computer vision technique can be used to conquer these challenges in a very reliable way, because there are so many different ways that your camera images can get tricked, e.g., rain, snow, reflection, obstruction, etc.



## Advanced Lane Finding 

The goals of this project is find road lane lines in a video stream and overlay the lines with a computed polynomial curve. 
In order to achieve this goal, methods and classes were built to do the following:

Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
Apply a distortion correction to raw images.
Use color transforms, gradients (Sobel), to create a thresholded binary image.
Apply a perspective transform to rectify binary image ("birds-eye view").
Detect lane pixels and fit to find the lane boundary.
Determine the curvature of the lane and vehicle position with respect to center.
Warp the detected lane boundaries back onto the original image.
Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

This jupyter notebook breaks down all the steps and display the output from each step. There is a separate pipeline written in notebook that combines all the steps with the same configuration and applies averaging techniques for the test video.


```python
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 
import glob
import math
from skimage.feature import corner_harris,corner_peaks
from moviepy.editor import VideoFileClip
from IPython.display import HTML
%matplotlib qt
%matplotlib inline

```

## Performing camera calibration 

Camera calibration is performed in order to correct the deformation in the images that is caused to the optic lens curvature. The first step is to print a chessboard and take random pictures of it. Then count the chess intersecting squires to provide "objp" which holds the (x,y,z) coordinates of these corners. Z=0 here and the object points are the same for all images in the calibration folder. The objpoints will be appended in "objp"  every time the method successfully detect all chessboard corners in a test image. "imgpoints" will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

"objpoints" and "imgpoints" were used to compute the camera calibration and distortion coefficients using the "cv2.calibrateCamera()" function on a test image in "cv2.undistort()" 


```python
# prepare object points. The number of corners are 6x9
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.
# Make a list of calibration images, all located in camera_cal
images = glob.glob('camera_cal/calibration*.jpg')
# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    # imread reads images in BGR format
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)
        #Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)
cv2.destroyAllWindows()
```

## Perform distortion removal on test images 
### 1. Has the distortion correction been correctly applied to each image?

Undistortion is performed on the provided test images before they are used in the pipeline. This also applies to the video frames. "dst" holds undistorted frames from "cv2.undistort" that were computed using "mtx".


```python
#Implement calibration on the images that will be used
def undistort(img, read=True, display=True, write=False):

# Test undistortion on an image
    
    if read:
        img = cv2.imread(img)
    img_size = (img.shape[1], img.shape[0])
#img = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
#dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
    if write:
        cv2.imwrite('test_images/test6.jpg',dst)
# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
#dist_pickle = {}
#dist_pickle["mtx"] = mtx
#dist_pickle["dist"] = dist
#pickle.dump( dist_pickle, open( "calibration_wide/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        img_RGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax1.imshow(img_RGB)
        ax1.set_title('Original Image', fontsize=30)
        dst_RGB=cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
        ax2.imshow(dst_RGB)
        ax2.set_title('Undistorted Image', fontsize=30)
    else:
        return dst
```


```python

```

## Image preprocessing and filtering
### 2. Has a binary image been created using color transforms, gradients or other methods?
Before finding the lane lines in the images, it's very important to perform further preprocessing. It's also helpful to take advantage of some properties found in 3 channel images such as color segmentation. I've tested 6 image filtering methods

1- Gaussian blurring

2- Gradients threshold

3- RGB splitting and thresholding

4- HLS splitting and thresholding

5- Magnitude thresholding

6- Edge detection (Sobel)

1- "cv2.GaussianBlur"  was applied to smooth all images and the images where converted to gray scale using "cv2.COLOR_RGB2GRAY". 
2- The thresholded binary image is finally shown in the figures below. Several thresholding ranges were tested and finally I determined that the range "thresh = (190, 255)" is bust suited for these images.


```python
def gaussian_noise(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255   
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def inverse_region_of_interest(img, vertices):
    """
    Applies an image mask.
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255   
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_or(mask, img)
    inverse_masked_image = cv2.bitwise_not(masked_image , img)
     
    #return masked_image
    return inverse_masked_image

def transform(img):
    imshape = img.shape
    img_size = (img.shape[1], img.shape[0])
  
    
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    
    
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    wraped =  cv2.warpPerspective(img,M,img_size, flags=cv2.INTER_LINEAR)
    
    return  Minv, wraped

```


```python
#test_image = cv2.imread('Frames/scene00161.jpg')
#Minv, warped = transform(test_image)
#plt.imshow(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
 
```


```python
#Read in the image
def gradients(img, read=True, display=True,  thresh = (190, 255)):
#image = mpimg.imread('Frames/scene00181.jpg')
    if read:
        image = undistort(img, display = False)
        #image = mpimg.imread(img)
#Blur the image
    blur_kernel_size = 1
    image = gaussian_noise(image, blur_kernel_size)
#Define a mask but only implement it after edge detection dot not be detected
    imshape = image.shape
        #vertices = np.array([[(80,imshape[0]),(400, 330), (580, 330), (imshape[1],imshape[0])]], dtype=np.int32)
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
#vertices = np.array([[(160,imshape[0]),(imshape[1]/2-60, imshape[0]/2+90),
                  #(imshape[1]/2+100, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
#image = region_of_interest(image, vertices)
   
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    binary = np.zeros_like(gray)
    binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(binary, cmap='gray')
        ax2.set_title('Thresholded Gradient', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    else: 
        return binary

```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    gradients(img)
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)



![png](output_11_5.png)



![png](output_11_6.png)



![png](output_11_7.png)



![png](output_11_8.png)



![png](output_11_9.png)



![png](output_11_10.png)



![png](output_11_11.png)



![png](output_11_12.png)



![png](output_11_13.png)



![png](output_11_14.png)



```python
#vertices_small = np.array([[(300,imshape[0]),(imshape[1]/2-110, imshape[0]/2+200),
 #                 (imshape[1]/2+200, imshape[0]/2+200), (imshape[1]-150,imshape[0])]], dtype=np.int32)

 
```

3- I've separated the RGB channels and applied a threshold of "thresh = (220, 255)" to remove the noise and fine the lane lines in the images. I also applied a mask defined by "vertices" to select the lane lines and remove everything else in the image.  
I've determined that the color channel "R" is best in finding the lane lines.


```python

def RGB_img(image,read = True, display = True, thresh = () ):
    #vertices_small = np.array([[(300,imshape[0]),(imshape[1]/2-110, imshape[0]/2+200),
     #                 (imshape[1]/2+200, imshape[0]/2+200), (imshape[1]-150,imshape[0])]], dtype=np.int32)
    if read:
        image = undistort(image, display = False)    
# Splitting RGB Channels
    R = image[:,:,0]
    G = image[:,:,1]
    B = image[:,:,2]
    
    imshape = image.shape
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
    #vertices = np.float32([[(490, 482),(810, 482),(1250, 720),(40, 720)]], dtype=np.int32)
    binary_R = np.zeros_like(R)
    binary_R[(R > thresh[0]) & (R <= thresh[1])] = 1
    binary_R= region_of_interest(binary_R, vertices)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)

    binary_G = np.zeros_like(G)
    binary_G[(G > thresh[0]) & (G <= thresh[1])] = 1
    binary_G= region_of_interest(binary_G, vertices)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)

    binary_B = np.zeros_like(B)
    binary_B[(B > thresh[0]) & (B <= thresh[1])] = 1
    binary_B= region_of_interest(binary_B, vertices)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(binary_R, cmap='gray')
        ax1.set_title('R', fontsize=50)
        ax2.imshow(binary_G, cmap='gray')
        ax2.set_title('G', fontsize=50)
        ax3.imshow(binary_B, cmap='gray')
        ax3.set_title('B', fontsize=50)
    else: 
        return binary_R, binary_G, binary_B
```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    RGB_img(img, thresh =(220,255) )
```


![png](output_15_0.png)



![png](output_15_1.png)



![png](output_15_2.png)



![png](output_15_3.png)



![png](output_15_4.png)



![png](output_15_5.png)



![png](output_15_6.png)



![png](output_15_7.png)



![png](output_15_8.png)



![png](output_15_9.png)



![png](output_15_10.png)



![png](output_15_11.png)



![png](output_15_12.png)



![png](output_15_13.png)



![png](output_15_14.png)

4- After separating the HLS channels and applying the same mask channel "S" was found to be the best for finding the lane lines. Several thresholds were tested and the rand "thresh = (150, 255)" was best in finding the lanes.

```python

def HLS_img(image,read = True, display = True, thresh = ()):
    #vertices_small = np.array([[(300,imshape[0]),(imshape[1]/2-110, imshape[0]/2+200),
     #                 (imshape[1]/2+200, imshape[0]/2+200), (imshape[1]-150,imshape[0])]], dtype=np.int32)
    if read:
        image = undistort(image, display = False)    
# Splitting RGB Channels
    H = image[:,:,0]
    L = image[:,:,1]
    S = image[:,:,2]
    
    imshape = image.shape
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
 
    binary_H = np.zeros_like(H)
    binary_H[(H > thresh[0]) & (H <= thresh[1])] = 1
    binary_H= region_of_interest(binary_H, vertices)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)

    binary_L = np.zeros_like(L)
    binary_L[(L > thresh[0]) & (L <= thresh[1])] = 1
    binary_L= region_of_interest(binary_L, vertices)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)

    binary_S = np.zeros_like(S)
    binary_S[(S > thresh[0]) & (S <= thresh[1])] = 1
    binary_S= region_of_interest(binary_S, vertices)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(binary_H, cmap='gray')
        ax1.set_title('H', fontsize=50)
        ax2.imshow(binary_L, cmap='gray')
        ax2.set_title('L', fontsize=50)
        ax3.imshow(binary_S, cmap='gray')
        ax3.set_title('S', fontsize=50)
    else: 
        return binary_H, binary_L, binary_S
```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    HLS_img(img, thresh=(200,255))
```


![png](output_18_0.png)



![png](output_18_1.png)



![png](output_18_2.png)



![png](output_18_3.png)



![png](output_18_4.png)



![png](output_18_5.png)



![png](output_18_6.png)



![png](output_18_7.png)



![png](output_18_8.png)



![png](output_18_9.png)



![png](output_18_10.png)



![png](output_18_11.png)



![png](output_18_12.png)



![png](output_18_13.png)



![png](output_18_14.png)



```python

def Lab_img(image,read = True, display = True, thresh = ()):
    #vertices_small = np.array([[(300,imshape[0]),(imshape[1]/2-110, imshape[0]/2+200),
     #                 (imshape[1]/2+200, imshape[0]/2+200), (imshape[1]-150,imshape[0])]], dtype=np.int32)
    if read:
        image = undistort(image, display = False)    
# Splitting RGB Channels
    blur_kernel_size = 1
    image = gaussian_noise(image, blur_kernel_size)
    L = image[:,:,0]
    a = image[:,:,1]
    b = image[:,:,2]
    
    imshape = image.shape
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
 
    binary_L = np.zeros_like(L)
    binary_L[(L > thresh[0]) & (L <= thresh[1])] = 1
    binary_L= region_of_interest(binary_L, vertices)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)
#binary_R= inverse_region_of_interest(binary_R, vertices_small)

    binary_a = np.zeros_like(a)
    binary_a[(a > thresh[0]) & (a <= thresh[1])] = 1
    binary_a= region_of_interest(binary_a, vertices)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)
#binary_G= inverse_region_of_interest(binary_G, vertices_small)

    binary_b = np.zeros_like(b)
    binary_b[(b > thresh[0]) & (b <= thresh[1])] = 1
    binary_b= region_of_interest(binary_b, vertices)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
#binary_B= inverse_region_of_interest(binary_B, vertices_small)
    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(binary_L, cmap='gray')
        ax1.set_title('L', fontsize=50)
        ax2.imshow(binary_a, cmap='gray')
        ax2.set_title('a', fontsize=50)
        ax3.imshow(binary_b, cmap='gray')
        ax3.set_title('b', fontsize=50)
    else: 
        return binary_L, binary_a, binary_b
```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    Lab_img(img, thresh=(200,255))
```


![png](output_20_0.png)



![png](output_20_1.png)



![png](output_20_2.png)



![png](output_20_3.png)



![png](output_20_4.png)



![png](output_20_5.png)



![png](output_20_6.png)



![png](output_20_7.png)



![png](output_20_8.png)



![png](output_20_9.png)



![png](output_20_10.png)



![png](output_20_11.png)



![png](output_20_12.png)



![png](output_20_13.png)



![png](output_20_14.png)


5- Edge detection is a widely used method for finding features in images. Here I used Sobel edge detection "cv2.Sobel" using "thresh=(50, 100)" on the thresholded "S" and "R" images as well as the gradient and magnitude threshold images. The figure shows the output where Sobel filter and R channel "Sobel_binary_S" is the best in defining the lane lines.   


```python
## return sobel threshold
def abs_sobel_thresh(img, orient, sobel_kernel, sobel_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1
    #binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

## return mag_direction

def mag_thresh(img, sobel_kernel, mag_thresh):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output

## return the gradient

def dir_threshold(img, sobel_kernel, thresh):
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx=np.absolute(sobelx)
    abs_sobely=np.absolute(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient 
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(dir_grad)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    #binary_output = np.copy(img) # Remove this line
    return binary_output
```


```python
def edge_detectors(img, display = True, read = True):

    if read:
        image = undistort(img, display = False) 
    blur_kernel_size = 1
    image = gaussian_noise(image, blur_kernel_size)
    imshape = image.shape
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)   
    ksize=3
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, sobel_thresh=(20,100))
    gradx = region_of_interest(gradx, vertices)
    #gradx= inverse_region_of_interest(gradx, vertices_small)
    #gradx= inverse_region_of_interest(gradx, vertices_small)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, sobel_thresh=(20,100))
    grady = region_of_interest(grady, vertices)
    #grady= inverse_region_of_interest(grady, vertices_small)
    #grady= inverse_region_of_interest(grady, vertices_small)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=(30,100))
    mag_binary = region_of_interest(mag_binary, vertices)
    #mag_binary= inverse_region_of_interest(mag_binary, vertices_small)
    #mag_binary= inverse_region_of_interest(mag_binary, vertices_small)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=(100, np.pi/2)) 
    dir_binary = region_of_interest(dir_binary, vertices)
    #dir_binary= inverse_region_of_interest(dir_binary, vertices_small)
    #dir_binary= inverse_region_of_interest(dir_binary, vertices_small)
    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(gradx, cmap='gray')
        ax1.set_title('SobelX', fontsize=50)
        ax2.imshow(mag_binary, cmap='gray')
        ax2.set_title('Magnitude', fontsize=50)
        ax3.imshow(dir_binary, cmap='gray')
        ax3.set_title('Gradient', fontsize=50)
    else: 
        return gradx, mag_binary, dir_binary
    

```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    edge_detectors(img)
```


![png](output_24_0.png)



![png](output_24_1.png)



![png](output_24_2.png)



![png](output_24_3.png)



![png](output_24_4.png)



![png](output_24_5.png)



![png](output_24_6.png)



![png](output_24_7.png)



![png](output_24_8.png)



![png](output_24_9.png)



![png](output_24_10.png)



![png](output_24_11.png)



![png](output_24_12.png)



![png](output_24_13.png)



![png](output_24_14.png)



```python
def combined_imgs(img, display = True):
      
    
    gradx, mag_binary, dir_binary = edge_detectors(img, display = False, read = True)
    binary_H, binary_L, binary_S = HLS_img(img,read = True, display = False, thresh = (230, 255))
    binary_R, binary_G, binary_B = RGB_img(img,read = True, display = False, thresh = (220, 255))
    binary_L, binary_a, binary_b = Lab_img(img,read = True, display = False, thresh = (200, 255))
    
    
    combined_B_b = np.zeros_like(binary_B)
    combined_B_b[(binary_B== 1) | (binary_b == 1)] = 1
    
    combined_S_b = np.zeros_like(binary_S)
    combined_S_b[(binary_S== 1) | (binary_b == 1)] = 1
    
    combined_S_B = np.zeros_like(binary_S)
    combined_S_B[(binary_S== 1) | (binary_B == 1)] = 1    
    
    if display:
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(combined_B_b, cmap='gray')
        ax1.set_title('combined_B_b', fontsize=50)
        ax2.imshow(combined_S_b, cmap='gray')
        ax2.set_title('combined_S_b', fontsize=50)
        ax3.imshow(combined_S_B, cmap='gray')
        ax3.set_title('combined_S_B', fontsize=50)
    else: 
        return combined_B_b, combined_S_b, combined_S_B    
```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    combined_imgs(img)
```


![png](output_26_0.png)



![png](output_26_1.png)



![png](output_26_2.png)



![png](output_26_3.png)



![png](output_26_4.png)



![png](output_26_5.png)



![png](output_26_6.png)



![png](output_26_7.png)



![png](output_26_8.png)



![png](output_26_9.png)



![png](output_26_10.png)



![png](output_26_11.png)



![png](output_26_12.png)



![png](output_26_13.png)



![png](output_26_14.png)


## Perspective transform
### 3. Has a perspective transform been applied to rectify the image?
Image perceptive transform is the process with which the image is warped in order to be displayed from a different perspective. This process tuned useful in finding the lanelines. "cv2.getPerspectiveTransform" is used for performing the transform by providing the source and destination coordinates "src" and "dst". The figures below show the orginal binary image and the transformed one.

 


```python
def warp(img, display = True):
    
    combined_B_b, combined_S_b, combined_S_B = combined_imgs(img, display = False)
    Minv, warped_img= transform(combined_S_B)
    histogram = np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))

        #plt.plot(histogram)
        ax1.plot( histogram)
        ax1.set_title('Histogram', fontsize=50)
        ax2.imshow(warped_img, cmap='gray')
        ax2.set_title('Warped Binary', fontsize=50)
    else:
        return warped_img
        
 
```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    warp(img)
```


![png](output_29_0.png)



![png](output_29_1.png)



![png](output_29_2.png)



![png](output_29_3.png)



![png](output_29_4.png)



![png](output_29_5.png)



![png](output_29_6.png)



![png](output_29_7.png)



![png](output_29_8.png)



![png](output_29_9.png)



![png](output_29_10.png)



![png](output_29_11.png)



![png](output_29_12.png)



![png](output_29_13.png)



![png](output_29_14.png)


## Locating the Lane Lines and Determine the Curvature

I have explored several methods for locating the lane lines in the images including

1- finding the histogram and applying a moving window to extract the lane lines.

2- finding the corners in the binary images and then covering them into arrays of coordinates

I have implemented a method for finding the histogram by using "np.sum(warped_img[warped_img.shape[0]/2:,:], axis=0)". I will not explain the approach followed for extracting the landlines from the histogram because I have ruled out this approach and used the second method. This decision was made after testing both methods and comparing the results. The code for finding the lanes using the histogram method was commented out below.

 

In the second approach I used Open CV "corner_peaks" and "corner_harris" to find the coordinates of detected lanes in the binary image. This method is better than the histogram method because is less computation demanding, faster, and more accurate in finding the lane lines. It also has "min_distance" attribute that gives me control over the smoothness of found lane lines and reducing noise from other detected objects. 
In order to identify the left and right lane lines, I applied a mask to divide the image into two equal regions and then computed "Harris" corner finding on each part separately. "corners_l" and "corners_r" are the two arrays containing the tuples for (x,y) coordinates in the left and right images. 


```python
"""
## define the left and right lane 
#Find the corners in the image and find their x and y coordinates. 
#warped_img = lines
row_w,col_w=warped_img.shape
warped_img_left=warped_img[0:row_w,0:math.ceil(col_w/2)]
warped_img_right=warped_img[0:row_w,math.ceil(col_w/2):col_w]
def show_corners(corners_l, corners_r,image,title=None):
    #Display a list of corners overlapping an image
    fig = plt.figure()
    plt.imshow(image,cmap='gray')
    y_corner_l,x_corner_l = zip(*corners_l)
    y_corner_r,x_corner_r = zip(*corners_r)
    plt.plot(x_corner_l,y_corner_l,'ro') # Plot corners
    plt.plot(x_corner_r,y_corner_r,'bo') # Plot corners
    if title:
        plt.title(title)
    plt.xlim(0,image.shape[1])
    plt.ylim(image.shape[0],0) # Images use weird axes
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()
    print ("Number of left corners:",len(corners_l),"Number of right corners:",len(corners_r), )    
corners_left = corner_peaks(corner_harris(warped_img_left),min_distance=15)
corners_right = corner_peaks(corner_harris(warped_img_right),min_distance=15)
show_corners(corners_left,corners_right+[0,math.ceil(col_w/2)],warped_img,title="corners found")    

"""
    
 
```




    '\n## define the left and right lane \n#Find the corners in the image and find their x and y coordinates. \n#warped_img = lines\nrow_w,col_w=warped_img.shape\nwarped_img_left=warped_img[0:row_w,0:math.ceil(col_w/2)]\nwarped_img_right=warped_img[0:row_w,math.ceil(col_w/2):col_w]\ndef show_corners(corners_l, corners_r,image,title=None):\n    #Display a list of corners overlapping an image\n    fig = plt.figure()\n    plt.imshow(image,cmap=\'gray\')\n    y_corner_l,x_corner_l = zip(*corners_l)\n    y_corner_r,x_corner_r = zip(*corners_r)\n    plt.plot(x_corner_l,y_corner_l,\'ro\') # Plot corners\n    plt.plot(x_corner_r,y_corner_r,\'bo\') # Plot corners\n    if title:\n        plt.title(title)\n    plt.xlim(0,image.shape[1])\n    plt.ylim(image.shape[0],0) # Images use weird axes\n    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)\n    plt.show()\n    print ("Number of left corners:",len(corners_l),"Number of right corners:",len(corners_r), )    \ncorners_left = corner_peaks(corner_harris(warped_img_left),min_distance=15)\ncorners_right = corner_peaks(corner_harris(warped_img_right),min_distance=15)\nshow_corners(corners_left,corners_right+[0,math.ceil(col_w/2)],warped_img,title="corners found")    \n\n'



## Radius of curvature finding, interpolation and extrapolation

### 4. Have lane line pixels been identified in the rectified image and fit with a polynomial?

A second order polynomial is used to find the lane lines as accurately as possible with "polyfit". However, the output line extends only to the detected lane lines and sometimes doesn't pass through the entire masked image. Therefore the line is extrapolated by defining the min and max points in the y direction. The figure below shown a plot of the identified corners and the polynomial passing through them.
To find the curvature I've estimated 30/720  meters per pixel in y dimension and 3.7/700  meters per pixel in x dimension using the recommendations from the course materials.  Based on that I have also estimated the position of the car using “center = abs(640 – ((rightx_int+leftx_int)/2)*3.7/700)”. 
Further smoothing was carried out by averaging the polynomials in the pipeline that was used to output the video.


```python
def curve(warped_img, display = True, pipeline = True):
    
    #image = cv2.imread(img)
    if pipeline:
        warped_img = warp(warped_img, display = False)
    
#y_corner_l,x_corner_l = zip(*corners_left)
#adjusted_corners_right= corners_right+[0,math.ceil(col_w/2)]                                    
#y_corner_r,x_corner_r = zip(*adjusted_corners_right)
#Measuring Curvature
#yvalus = warped_img.shape[0]
#Represent lane-line pixels
#leftx = np.array(x_corner_l)
#lefty = np.array(y_corner_l)
#rightx = np.array(x_corner_r)
#righty = np.array(y_corner_r)
#########################################################
    rightx = []
    righty = []
    leftx = []
    lefty = []
    x, y = np.nonzero(np.transpose(warped_img))
    row_w,col_w=warped_img.shape
    col_w = math.floor(col_w/2)
    i = row_w
    j = col_w
    offset=25
    angle=90
    while j >= 0:
        histogram = np.sum(warped_img[j:i,:], axis=0)
        left_peak = np.argmax(histogram[:col_w])
        x_idx = np.where((((left_peak - offset) < x)&(x < (left_peak + offset))&((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            leftx.extend(x_window.tolist())
            lefty.extend(y_window.tolist())
        right_peak = np.argmax(histogram[col_w:]) + col_w
        x_idx = np.where((((right_peak - offset) < x)&(x < (right_peak + offset))&((y > j) & (y < i))))
        x_window, y_window = x[x_idx], y[x_idx]
        if np.sum(x_window) != 0:
            rightx.extend(x_window.tolist())
            righty.extend(y_window.tolist())
        i -= angle
        j -= angle

    leftx = np.array(leftx)
    lefty = np.array(lefty)
    rightx = np.array(rightx)
    righty = np.array(righty)
    ###################################################
    # Fit a second order polynomial to each line
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    #Extrapolation fit the line in top and bottom
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    rightx_int = right_fit[0]*row_w**2 + right_fit[1]*row_w + right_fit[2]
    rightx = np.append(rightx,rightx_int)
    righty = np.append(righty, row_w)
    rightx = np.append(rightx,right_fit[0]*0**2 + right_fit[1]*0 + right_fit[2])
    righty = np.append(righty, 0)
    leftx_int = left_fit[0]*row_w**2 + left_fit[1]*row_w + left_fit[2]
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, row_w)
    leftx = np.append(leftx,left_fit[0]*0**2 + left_fit[1]*0 + left_fit[2])
    lefty = np.append(lefty, 0)
    lsort = np.argsort(lefty)
    rsort = np.argsort(righty)
    lefty = lefty[lsort]
    leftx = leftx[lsort]
    righty = righty[rsort]
    rightx = rightx[rsort]
    left_fit = np.polyfit(lefty, leftx, 2)
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    right_fit = np.polyfit(righty, rightx, 2)
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    ########

    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_l = np.max(row_w)
    y_eval_r = np.max(row_w)
    left_curverad = ((1 + (2*left_fit[0]*y_eval_l + left_fit[1])**2)**1.5) \
                                 /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval_r + right_fit[1])**2)**1.5) \
                                    /np.absolute(2*right_fit[0])
    

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/row_w # meters per pixel in y dimension
    xm_per_pix = 3.7/row_w # meteres per pixel in x dimension

    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_l + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_r + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    center = abs((col_w - ((rightx_int+leftx_int)/2))*3.7/row_w)    
              # Calculate the position of the vehicle
    ## Radius of curvature finding, interpolation and extrapolation

                  # Calculate the position of the vehicle

    #img_size = (image.shape[1], image.shape[0])    
        # Now our radius of curvature is in meters
    #This part is for visualization, import images without binarization
    # undistort and wrap then overlay the detected lines
    image = undistort(img, read=True, display=False, write=False)
    imshape = image.shape
    vertices = np.array([[(160,imshape[0]),(imshape[1]/2-70, imshape[0]/2+90),
                      (imshape[1]/2+130, imshape[0]/2+90), (imshape[1]-20,imshape[0])]], dtype=np.int32)
    
          
    #orginal= region_of_interest(image, vertices)
    Minv, warped_orginal = transform(image)
    ##########################
    #Drawing the lines back down onto the road
    #mage called warped, and you have fit the lines with a polynomial and have arrays called yvals, 
    #left_fitx and right_fitx, 
    img_size = (image.shape[1], image.shape[0])
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, lefty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, righty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    final_result=cv2.putText(result, "Left:{}".format(math.floor(left_curverad)),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    final_result=cv2.putText(result, "Right:{}".format(math.floor(right_curverad)),(10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    if center < 640:
            final_result=cv2.putText(result, "Car is {:.2f}m left of center".format(center),(10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    else:
            final_result=cv2.putText(result, "Car is {:.2f}m right of center".format(center),(10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    #plt.imshow(final_result)
########################################
    


    
    
    ###
    if display:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))   
        f.tight_layout()
        ax1.imshow(cv2.cvtColor(warped_orginal, cv2.COLOR_BGR2RGB))
        ax1.plot(leftx, lefty, 'o', color='red')
        ax1.plot(rightx, righty, 'o', color='blue')
        ax1.set_xlim(0, col_w*2)
        ax1.set_ylim(0, row_w)
        ax1.plot(left_fitx, lefty, color='green', linewidth=3)
        ax1.plot(right_fitx, righty, color='green', linewidth=3)
        ax1.invert_yaxis()
        ax2.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
    else:
        return  final_result
        
    
  

```


```python
imgs = glob.glob('Frames/*.jpg')
for img in imgs:
    curve(img)
```


![png](output_35_0.png)



![png](output_35_1.png)



![png](output_35_2.png)



![png](output_35_3.png)



![png](output_35_4.png)



![png](output_35_5.png)



![png](output_35_6.png)



![png](output_35_7.png)



![png](output_35_8.png)



![png](output_35_9.png)



![png](output_35_10.png)



![png](output_35_11.png)



![png](output_35_12.png)



![png](output_35_13.png)



![png](output_35_14.png)


### 5. Having identified the lane lines, has the radius of curvature of the road been estimated? And the position of the vehicle with respect to center in the lane?

Yes, the results are shown in the image below.


```python
##final pipeline
class Left:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False
        
        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None
        
        # Store recent x intercepts for averaging across frames
        self.x_int = []
        self.top = []
        
        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None
        
        # Remember radius of curvature
        self.radius = None
        
        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = []
        self.fit1 = []
        self.fit2 = []
        self.fitx = None
        self.pts = []
        
        # Count the number of frames
        self.count = 0

        
class Right:
    def __init__(self):
        # Was the line found in the previous frame?
        self.found = False
        
        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None
        
        # Store recent x intercepts for averaging across frames
        self.x_int = []
        self.top = []
        
        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None
        
        # Remember radius of curvature
        self.radius = None
        
        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = []
        self.fit1 = []
        self.fit2 = []
        self.fitx = None
        self.pts = []

def video_pipeline (image):
    #Step 1 undistort the image based on the camera calibration
    #image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = undistort(image, read=False, display=False, write=False)
    #Step 2 Blur the image
    blur_kernel_size = 1
    image = gaussian_noise(image, blur_kernel_size)
    #Step 3 return RGB, HLS, and Lab image
    binary_R, binary_G, binary_B = RGB_img(image,read = False, display = False, thresh = (220,255))
    binary_H, binary_L, binary_S = HLS_img(image,read = False, display = False, thresh = (200,255))
    binary_L, binary_a, binary_b = Lab_img(image,read = False, display = False, thresh = (200,255))
    #Step 4 return combined images
    combined_B_b = np.zeros_like(binary_B)
    combined_B_b[(binary_B== 1) | (binary_b == 1)] = 1
    
    combined_S_b = np.zeros_like(binary_S)
    combined_S_b[(binary_S== 1) | (binary_b == 1)] = 1
    
    combined_S_B = np.zeros_like(binary_S)
    combined_S_B[(binary_S== 1) | (binary_B == 1)] = 1 
    #Step 5 Warp image 
    Minv, warped_img= transform(combined_S_B)
     
    #Step 6, line fitting
    #final_image = curve(warped_img, display = False, pipeline = False)
    rightx = []
    righty = []
    leftx = []
    lefty = []
    angle = 90
    # Find pixels in the binnary image
    x, y = np.nonzero(np.transpose(warped_img)) 

    if Left.found == True: # Search for left lane pixels around previous polynomial
        i = 720
        j = 630
        while j >= 0:
            yval = np.mean([i,j])
            xval = (np.mean(Left.fit0))*yval**2 + (np.mean(Left.fit1))*yval + (np.mean(Left.fit2))
            x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:
                np.append(leftx, x_window)
                np.append(lefty, y_window)
            i -= angle
            j -= angle
        if np.sum(leftx) == 0: 
            Left.found = False # If no lane pixels were detected then perform blind search
        
    if Right.found == True: # Search for right lane pixels around previous polynomial
        i = 720
        j = 630
        while j >= 0:
            yval = np.mean([i,j])
            xval = (np.mean(Right.fit0))*yval**2 + (np.mean(Right.fit1))*yval + (np.mean(Right.fit2))
            x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:
                np.append(rightx, x_window)
                np.append(righty, y_window)
            
            i -= angle
            j -= angle
        if np.sum(rightx) == 0:
            Right.found = False # If no lane pixels were detected then perform blind search
            
    if Right.found == False: # Perform blind search for lane lines
        i = 720
        j = 630
        while j >= 0:
            histogram = np.sum(warped_img[j:i,:], axis=0)
            right_peak = np.argmax(histogram[640:]) + 640
            x_idx = np.where((((right_peak - 25) < x)&(x < (right_peak + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:
                rightx.extend(x_window.tolist())
                righty.extend(y_window.tolist())
            i -= angle
            j -= angle
    if not np.sum(righty) > 0:
        righty = Right.Y
        rightx = Right.X
            
    if Left.found == False:# Perform blind search for lane lines
        i = 720
        j = 630
        while j >= 0:
            histogram = np.sum(warped_img[j:i,:], axis=0)
            left_peak = np.argmax(histogram[:640])
            x_idx = np.where((((left_peak - 25) < x)&(x < (left_peak + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:
                leftx.extend(x_window.tolist())
                lefty.extend(y_window.tolist())
            i -= angle
            j -= angle
    if not np.sum(lefty) > 0:
        lefty = Left.Y
        leftx = Left.X
        
    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)     
    # Calculate left polynomial fit based on detected pixels
    left_fit = np.polyfit(lefty, leftx, 2)
    
    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int = left_fit[0]*720**2 + left_fit[1]*720 + left_fit[2]
    left_top = left_fit[0]*0**2 + left_fit[1]*0 + left_fit[2]
    
    # Average intercepts across 5 frames
    Left.x_int.append(leftx_int)
    Left.top.append(left_top)
    leftx_int = np.mean(Left.x_int)
    left_top = np.mean(Left.top)
    Left.lastx_int = leftx_int
    Left.last_top = left_top
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)
    lsort = np.argsort(lefty)
    lefty = lefty[lsort]
    leftx = leftx[lsort]
    Left.X = leftx
    Left.Y = lefty
    
    # Recalculate polynomial with intercepts and average across 5 frames
    left_fit = np.polyfit(lefty, leftx, 2)
    Left.fit0.append(left_fit[0])
    Left.fit1.append(left_fit[1])
    Left.fit2.append(left_fit[2])
    left_fit = [np.mean(Left.fit0), 
                np.mean(Left.fit1), 
                np.mean(Left.fit2)]
    
    # Fit polynomial to detected pixels
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    Left.fitx = left_fitx
    
    # Calculate right polynomial fit based on detected pixels
    right_fit = np.polyfit(np.int_(righty), np.int_(rightx), 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int = right_fit[0]*720**2 + right_fit[1]*720 + right_fit[2]
    right_top = right_fit[0]*0**2 + right_fit[1]*0 + right_fit[2]
    
    # Average intercepts across 5 frames
    Right.x_int.append(rightx_int)
    rightx_int = np.mean(Right.x_int)
    Right.top.append(right_top)
    right_top = np.mean(Right.top)
    Right.lastx_int = rightx_int
    Right.last_top = right_top
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)
    rsort = np.argsort(righty)
    righty = righty[rsort]
    rightx = rightx[rsort]
    Right.X = rightx
    Right.Y = righty
    
    # Recalculate polynomial with intercepts and average across 5 frames
    right_fit = np.polyfit(righty, rightx, 2)
    Right.fit0.append(right_fit[0])
    Right.fit1.append(right_fit[1])
    Right.fit2.append(right_fit[2])
    right_fit = [np.mean(Right.fit0), np.mean(Right.fit1), np.mean(Right.fit2)]
    
    # Fit polynomial to detected pixels
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    Right.fitx = right_fitx
        
    # Compute radius of curvature for each lane in meters
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
        
    
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left.fitx, Left.Y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, Right.Y]))])
    pts = np.hstack((pts_left, pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_(pts), (34,255,34))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.5, 0)
    
    
    if len(Left.fit0) > 10:
        Left.fit0 = Left.fit0[1:]
    if len(Left.fit1) > 10:
        Left.fit1 = Left.fit1[1:]
    if len(Left.fit2) > 10:
        Left.fit2 = Left.fit2[1:]
    if len(Left.x_int) > 50:
        Left.x_int = Left.x_int[1:]
    if len(Left.top) > 50:
        Left.top = Left.top[1:]
    if len(Right.fit0) > 10:
        Right.fit0 = Right.fit0[1:]
    if len(Right.fit1) > 10:
        Right.fit1 = Right.fit1[1:]
    if len(Right.fit2) > 10:
        Right.fit2 = Right.fit2[1:]
    if len(Right.x_int) > 50:
        Right.x_int = Right.x_int[1:]
    if len(Right.top) > 50:
        Right.top = Right.top[1:]
        
    final_result=cv2.putText(result, "Left:{}".format(math.floor(left_curverad)),(10, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
    final_result=cv2.putText(result, "Right:{}".format(math.floor(right_curverad)),(10, 70), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
    if distance_from_center < 640:
            final_result=cv2.putText(result, "Car is {:.2f}m left of center".format(distance_from_center),(10, 110), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)
    else:
            final_result=cv2.putText(result, "Car is {:.2f}m right of center".format(distance_from_center),(10, 110), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 255, 0), 3)    
   
    Left.count += 1
    return result
    
```


```python
Left.__init__(Left)
Right.__init__(Right)
video_output = 'challange_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
white_clip = clip1.fl_image(video_pipeline) 
white_clip.write_videofile(video_output, audio=False)
```

    [MoviePy] >>>> Building video challange_output.mp4
    [MoviePy] Writing video challange_output.mp4


    
      0%|          | 0/1261 [00:00<?, ?it/s][A
      0%|          | 1/1261 [00:01<31:28,  1.50s/it][A
      0%|          | 2/1261 [00:03<31:50,  1.52s/it][A
      0%|          | 3/1261 [00:04<31:51,  1.52s/it][A
      0%|          | 4/1261 [00:06<32:11,  1.54s/it][A
      0%|          | 5/1261 [00:07<32:04,  1.53s/it][A
      0%|          | 6/1261 [00:09<31:56,  1.53s/it][A
      1%|          | 7/1261 [00:10<31:40,  1.52s/it][A
      1%|          | 8/1261 [00:12<31:39,  1.52s/it][A
      1%|          | 9/1261 [00:13<31:44,  1.52s/it][A
      1%|          | 10/1261 [00:15<31:41,  1.52s/it][A
      1%|          | 11/1261 [00:16<31:45,  1.52s/it][A
      1%|          | 12/1261 [00:18<31:28,  1.51s/it][A
      1%|          | 13/1261 [00:19<31:34,  1.52s/it][A
      1%|          | 14/1261 [00:21<31:41,  1.52s/it][A
      1%|          | 15/1261 [00:22<31:49,  1.53s/it][A
      1%|▏         | 16/1261 [00:24<32:09,  1.55s/it][A
      1%|▏         | 17/1261 [00:26<32:19,  1.56s/it][A
      1%|▏         | 18/1261 [00:27<32:22,  1.56s/it][A
      2%|▏         | 19/1261 [00:29<32:19,  1.56s/it][A
      2%|▏         | 20/1261 [00:30<31:45,  1.54s/it][A
      2%|▏         | 21/1261 [00:32<31:41,  1.53s/it][A
      2%|▏         | 22/1261 [00:33<31:16,  1.51s/it][A
      2%|▏         | 23/1261 [00:35<31:08,  1.51s/it][A
      2%|▏         | 24/1261 [00:36<30:53,  1.50s/it][A
      2%|▏         | 25/1261 [00:38<31:20,  1.52s/it][A
      2%|▏         | 26/1261 [00:39<31:14,  1.52s/it][A
      2%|▏         | 27/1261 [00:41<31:18,  1.52s/it][A
      2%|▏         | 28/1261 [00:42<31:03,  1.51s/it][A
      2%|▏         | 29/1261 [00:44<31:20,  1.53s/it][A
      2%|▏         | 30/1261 [00:45<31:32,  1.54s/it][A
      2%|▏         | 31/1261 [00:47<32:01,  1.56s/it][A
      3%|▎         | 32/1261 [00:49<31:59,  1.56s/it][A
      3%|▎         | 33/1261 [00:50<31:39,  1.55s/it][A
      3%|▎         | 34/1261 [00:52<31:34,  1.54s/it][A
      3%|▎         | 35/1261 [00:53<31:13,  1.53s/it][A
      3%|▎         | 36/1261 [00:55<31:09,  1.53s/it][A
      3%|▎         | 37/1261 [00:56<30:51,  1.51s/it][A
      3%|▎         | 38/1261 [00:58<30:54,  1.52s/it][A
      3%|▎         | 39/1261 [00:59<30:57,  1.52s/it][A
      3%|▎         | 40/1261 [01:01<30:57,  1.52s/it][A
      3%|▎         | 41/1261 [01:02<30:59,  1.52s/it][A
      3%|▎         | 42/1261 [01:04<31:08,  1.53s/it][A
      3%|▎         | 43/1261 [01:05<31:24,  1.55s/it][A
      3%|▎         | 44/1261 [01:07<31:34,  1.56s/it][A
      4%|▎         | 45/1261 [01:08<31:33,  1.56s/it][A
      4%|▎         | 46/1261 [01:10<31:18,  1.55s/it][A
      4%|▎         | 47/1261 [01:12<31:16,  1.55s/it][A
      4%|▍         | 48/1261 [01:13<31:15,  1.55s/it][A
      4%|▍         | 49/1261 [01:15<30:57,  1.53s/it][A
      4%|▍         | 50/1261 [01:16<30:47,  1.53s/it][A
      4%|▍         | 51/1261 [01:18<31:17,  1.55s/it][A
      4%|▍         | 52/1261 [01:19<31:09,  1.55s/it][A
      4%|▍         | 53/1261 [01:21<31:27,  1.56s/it][A
      4%|▍         | 54/1261 [01:22<31:28,  1.56s/it][A
      4%|▍         | 55/1261 [01:24<31:33,  1.57s/it][A
      4%|▍         | 56/1261 [01:26<31:48,  1.58s/it][A
      5%|▍         | 57/1261 [01:27<31:28,  1.57s/it][A
      5%|▍         | 58/1261 [01:29<31:17,  1.56s/it][A
      5%|▍         | 59/1261 [01:30<30:52,  1.54s/it][A
      5%|▍         | 60/1261 [01:32<31:16,  1.56s/it][A
      5%|▍         | 61/1261 [01:33<30:52,  1.54s/it][A
      5%|▍         | 62/1261 [01:35<30:48,  1.54s/it][A
      5%|▍         | 63/1261 [01:36<30:47,  1.54s/it][A
      5%|▌         | 64/1261 [01:38<30:20,  1.52s/it][A
      5%|▌         | 65/1261 [01:39<30:31,  1.53s/it][A
      5%|▌         | 66/1261 [01:41<31:02,  1.56s/it][A
      5%|▌         | 67/1261 [01:43<31:25,  1.58s/it][A
      5%|▌         | 68/1261 [01:44<31:06,  1.56s/it][A
      5%|▌         | 69/1261 [01:46<30:59,  1.56s/it][A
      6%|▌         | 70/1261 [01:47<31:30,  1.59s/it][A
      6%|▌         | 71/1261 [01:49<32:07,  1.62s/it][A
      6%|▌         | 72/1261 [01:51<31:54,  1.61s/it][A
      6%|▌         | 73/1261 [01:52<31:32,  1.59s/it][A
      6%|▌         | 74/1261 [01:54<31:10,  1.58s/it][A
      6%|▌         | 75/1261 [01:55<31:03,  1.57s/it][A
      6%|▌         | 76/1261 [01:57<31:06,  1.57s/it][A
      6%|▌         | 77/1261 [01:58<31:00,  1.57s/it][A
      6%|▌         | 78/1261 [02:00<30:11,  1.53s/it][A
      6%|▋         | 79/1261 [02:01<30:16,  1.54s/it][A
      6%|▋         | 80/1261 [02:03<30:01,  1.53s/it][A
      6%|▋         | 81/1261 [02:05<30:29,  1.55s/it][A
      7%|▋         | 82/1261 [02:06<30:11,  1.54s/it][A
      7%|▋         | 83/1261 [02:08<30:25,  1.55s/it][A
      7%|▋         | 84/1261 [02:09<30:07,  1.54s/it][A
      7%|▋         | 85/1261 [02:11<30:13,  1.54s/it][A
      7%|▋         | 86/1261 [02:12<29:58,  1.53s/it][A
      7%|▋         | 87/1261 [02:14<30:02,  1.54s/it][A
      7%|▋         | 88/1261 [02:15<30:02,  1.54s/it][A
      7%|▋         | 89/1261 [02:17<29:54,  1.53s/it][A
      7%|▋         | 90/1261 [02:18<29:59,  1.54s/it][A
      7%|▋         | 91/1261 [02:20<29:48,  1.53s/it][A
      7%|▋         | 92/1261 [02:21<30:05,  1.54s/it][A
      7%|▋         | 93/1261 [02:23<30:14,  1.55s/it][A
      7%|▋         | 94/1261 [02:25<30:13,  1.55s/it][A
      8%|▊         | 95/1261 [02:26<29:49,  1.54s/it][A
      8%|▊         | 96/1261 [02:28<29:42,  1.53s/it][A
      8%|▊         | 97/1261 [02:29<29:16,  1.51s/it][A
      8%|▊         | 98/1261 [02:31<29:44,  1.53s/it][A
      8%|▊         | 99/1261 [02:32<29:33,  1.53s/it][A
      8%|▊         | 100/1261 [02:34<29:39,  1.53s/it][A
      8%|▊         | 101/1261 [02:35<30:38,  1.59s/it][A
      8%|▊         | 102/1261 [02:37<31:19,  1.62s/it][A
      8%|▊         | 103/1261 [02:39<31:23,  1.63s/it][A
      8%|▊         | 104/1261 [02:40<31:10,  1.62s/it][A
      8%|▊         | 105/1261 [02:42<31:12,  1.62s/it][A
      8%|▊         | 106/1261 [02:44<31:05,  1.62s/it][A
      8%|▊         | 107/1261 [02:45<31:05,  1.62s/it][A
      9%|▊         | 108/1261 [02:47<31:04,  1.62s/it][A
      9%|▊         | 109/1261 [02:48<30:46,  1.60s/it][A
      9%|▊         | 110/1261 [02:50<30:20,  1.58s/it][A
      9%|▉         | 111/1261 [02:51<30:00,  1.57s/it][A
      9%|▉         | 112/1261 [02:53<30:03,  1.57s/it][A
      9%|▉         | 113/1261 [02:55<30:01,  1.57s/it][A
      9%|▉         | 114/1261 [02:56<29:45,  1.56s/it][A
      9%|▉         | 115/1261 [02:58<29:50,  1.56s/it][A
      9%|▉         | 116/1261 [02:59<29:48,  1.56s/it][A
      9%|▉         | 117/1261 [03:01<29:50,  1.57s/it][A
      9%|▉         | 118/1261 [03:02<29:37,  1.55s/it][A
      9%|▉         | 119/1261 [03:04<29:40,  1.56s/it][A
     10%|▉         | 120/1261 [03:05<29:36,  1.56s/it][A
     10%|▉         | 121/1261 [03:07<29:28,  1.55s/it][A
     10%|▉         | 122/1261 [03:09<29:23,  1.55s/it][A
     10%|▉         | 123/1261 [03:10<28:57,  1.53s/it][A
     10%|▉         | 124/1261 [03:12<29:13,  1.54s/it][A
     10%|▉         | 125/1261 [03:13<28:57,  1.53s/it][A
     10%|▉         | 126/1261 [03:15<29:19,  1.55s/it][A
     10%|█         | 127/1261 [03:16<29:34,  1.56s/it][A
     10%|█         | 128/1261 [03:18<29:15,  1.55s/it][A
     10%|█         | 129/1261 [03:19<29:21,  1.56s/it][A
     10%|█         | 130/1261 [03:21<29:25,  1.56s/it][A
     10%|█         | 131/1261 [03:23<29:42,  1.58s/it][A
     10%|█         | 132/1261 [03:24<29:39,  1.58s/it][A
     11%|█         | 133/1261 [03:26<29:24,  1.56s/it][A
     11%|█         | 134/1261 [03:27<29:10,  1.55s/it][A
     11%|█         | 135/1261 [03:29<29:10,  1.55s/it][A
     11%|█         | 136/1261 [03:30<29:14,  1.56s/it][A
     11%|█         | 137/1261 [03:32<28:57,  1.55s/it][A
     11%|█         | 138/1261 [03:33<28:53,  1.54s/it][A
     11%|█         | 139/1261 [03:35<28:58,  1.55s/it][A
     11%|█         | 140/1261 [03:37<28:59,  1.55s/it][A
     11%|█         | 141/1261 [03:38<28:46,  1.54s/it][A
     11%|█▏        | 142/1261 [03:40<29:01,  1.56s/it][A
     11%|█▏        | 143/1261 [03:41<29:01,  1.56s/it][A
     11%|█▏        | 144/1261 [03:43<29:06,  1.56s/it][A
     11%|█▏        | 145/1261 [03:44<29:22,  1.58s/it][A
     12%|█▏        | 146/1261 [03:46<29:27,  1.59s/it][A
     12%|█▏        | 147/1261 [03:48<29:46,  1.60s/it][A
     12%|█▏        | 148/1261 [03:49<30:03,  1.62s/it][A
     12%|█▏        | 149/1261 [03:51<29:47,  1.61s/it][A
     12%|█▏        | 150/1261 [03:53<30:01,  1.62s/it][A
     12%|█▏        | 151/1261 [03:54<29:47,  1.61s/it][A
     12%|█▏        | 152/1261 [03:56<29:31,  1.60s/it][A
     12%|█▏        | 153/1261 [03:57<29:28,  1.60s/it][A
     12%|█▏        | 154/1261 [03:59<29:22,  1.59s/it][A
     12%|█▏        | 155/1261 [04:00<29:31,  1.60s/it][A
     12%|█▏        | 156/1261 [04:02<29:24,  1.60s/it][A
     12%|█▏        | 157/1261 [04:04<29:24,  1.60s/it][A
     13%|█▎        | 158/1261 [04:05<29:00,  1.58s/it][A
     13%|█▎        | 159/1261 [04:07<28:34,  1.56s/it][A
     13%|█▎        | 160/1261 [04:08<28:37,  1.56s/it][A
     13%|█▎        | 161/1261 [04:10<28:31,  1.56s/it][A
     13%|█▎        | 162/1261 [04:11<28:12,  1.54s/it][A
     13%|█▎        | 163/1261 [04:13<28:28,  1.56s/it][A
     13%|█▎        | 164/1261 [04:14<28:27,  1.56s/it][A
     13%|█▎        | 165/1261 [04:16<28:19,  1.55s/it][A
     13%|█▎        | 166/1261 [04:18<28:19,  1.55s/it][A
     13%|█▎        | 167/1261 [04:19<28:40,  1.57s/it][A
     13%|█▎        | 168/1261 [04:21<28:40,  1.57s/it][A
     13%|█▎        | 169/1261 [04:22<28:40,  1.58s/it][A
     13%|█▎        | 170/1261 [04:24<28:34,  1.57s/it][A
     14%|█▎        | 171/1261 [04:25<28:26,  1.57s/it][A
     14%|█▎        | 172/1261 [04:27<28:03,  1.55s/it][A
     14%|█▎        | 173/1261 [04:29<28:08,  1.55s/it][A
     14%|█▍        | 174/1261 [04:30<27:58,  1.54s/it][A
     14%|█▍        | 175/1261 [04:32<27:56,  1.54s/it][A
     14%|█▍        | 176/1261 [04:33<27:40,  1.53s/it][A
     14%|█▍        | 177/1261 [04:35<27:45,  1.54s/it][A
     14%|█▍        | 178/1261 [04:36<27:47,  1.54s/it][A
     14%|█▍        | 179/1261 [04:38<27:57,  1.55s/it][A
     14%|█▍        | 180/1261 [04:39<28:04,  1.56s/it][A
     14%|█▍        | 181/1261 [04:41<27:58,  1.55s/it][A
     14%|█▍        | 182/1261 [04:42<27:47,  1.55s/it][A
     15%|█▍        | 183/1261 [04:44<27:45,  1.54s/it][A
     15%|█▍        | 184/1261 [04:46<27:54,  1.55s/it][A
     15%|█▍        | 185/1261 [04:47<28:01,  1.56s/it][A
     15%|█▍        | 186/1261 [04:49<28:03,  1.57s/it][A
     15%|█▍        | 187/1261 [04:50<28:03,  1.57s/it][A
     15%|█▍        | 188/1261 [04:52<27:47,  1.55s/it][A
     15%|█▍        | 189/1261 [04:53<27:44,  1.55s/it][A
     15%|█▌        | 190/1261 [04:55<27:46,  1.56s/it][A
     15%|█▌        | 191/1261 [04:56<27:59,  1.57s/it][A
     15%|█▌        | 192/1261 [04:58<27:45,  1.56s/it][A
     15%|█▌        | 193/1261 [05:00<27:44,  1.56s/it][A
     15%|█▌        | 194/1261 [05:01<27:56,  1.57s/it][A
     15%|█▌        | 195/1261 [05:03<27:47,  1.56s/it][A
     16%|█▌        | 196/1261 [05:04<27:28,  1.55s/it][A
     16%|█▌        | 197/1261 [05:06<27:24,  1.55s/it][A
     16%|█▌        | 198/1261 [05:07<27:23,  1.55s/it][A
     16%|█▌        | 199/1261 [05:09<27:22,  1.55s/it][A
     16%|█▌        | 200/1261 [05:10<27:30,  1.56s/it][A
     16%|█▌        | 201/1261 [05:12<27:18,  1.55s/it][A
     16%|█▌        | 202/1261 [05:14<27:34,  1.56s/it][A
     16%|█▌        | 203/1261 [05:15<27:29,  1.56s/it][A
     16%|█▌        | 204/1261 [05:17<27:37,  1.57s/it][A
     16%|█▋        | 205/1261 [05:18<27:42,  1.57s/it][A
     16%|█▋        | 206/1261 [05:20<27:41,  1.58s/it][A
     16%|█▋        | 207/1261 [05:21<27:48,  1.58s/it][A
     16%|█▋        | 208/1261 [05:23<27:24,  1.56s/it][A
     17%|█▋        | 209/1261 [05:24<27:07,  1.55s/it][A
     17%|█▋        | 210/1261 [05:26<26:59,  1.54s/it][A
     17%|█▋        | 211/1261 [05:28<27:06,  1.55s/it][A
     17%|█▋        | 212/1261 [05:29<27:14,  1.56s/it][A
     17%|█▋        | 213/1261 [05:31<27:01,  1.55s/it][A
     17%|█▋        | 214/1261 [05:32<27:09,  1.56s/it][A
     17%|█▋        | 215/1261 [05:34<26:54,  1.54s/it][A
     17%|█▋        | 216/1261 [05:35<27:08,  1.56s/it][A
     17%|█▋        | 217/1261 [05:37<27:15,  1.57s/it][A
     17%|█▋        | 218/1261 [05:39<27:15,  1.57s/it][A
     17%|█▋        | 219/1261 [05:40<27:15,  1.57s/it][A
     17%|█▋        | 220/1261 [05:42<27:08,  1.56s/it][A
     18%|█▊        | 221/1261 [05:43<26:53,  1.55s/it][A
     18%|█▊        | 222/1261 [05:45<27:04,  1.56s/it][A
     18%|█▊        | 223/1261 [05:46<27:17,  1.58s/it][A
     18%|█▊        | 224/1261 [05:48<27:33,  1.59s/it][A
     18%|█▊        | 225/1261 [05:50<27:35,  1.60s/it][A
     18%|█▊        | 226/1261 [05:51<27:20,  1.59s/it][A
     18%|█▊        | 227/1261 [05:53<27:27,  1.59s/it][A
     18%|█▊        | 228/1261 [05:54<27:28,  1.60s/it][A
     18%|█▊        | 229/1261 [05:56<27:35,  1.60s/it][A
     18%|█▊        | 230/1261 [05:58<27:18,  1.59s/it][A
     18%|█▊        | 231/1261 [05:59<26:49,  1.56s/it][A
     18%|█▊        | 232/1261 [06:01<26:44,  1.56s/it][A
     18%|█▊        | 233/1261 [06:02<26:39,  1.56s/it][A
     19%|█▊        | 234/1261 [06:04<26:32,  1.55s/it][A
     19%|█▊        | 235/1261 [06:05<26:25,  1.55s/it][A
     19%|█▊        | 236/1261 [06:07<26:18,  1.54s/it][A
     19%|█▉        | 237/1261 [06:08<26:13,  1.54s/it][A
     19%|█▉        | 238/1261 [06:10<26:01,  1.53s/it][A
     19%|█▉        | 239/1261 [06:11<26:14,  1.54s/it][A
     19%|█▉        | 240/1261 [06:13<26:04,  1.53s/it][A
     19%|█▉        | 241/1261 [06:14<26:20,  1.55s/it][A
     19%|█▉        | 242/1261 [06:16<26:29,  1.56s/it][A
     19%|█▉        | 243/1261 [06:18<26:23,  1.56s/it][A
     19%|█▉        | 244/1261 [06:19<26:01,  1.54s/it][A
     19%|█▉        | 245/1261 [06:21<25:55,  1.53s/it][A
     20%|█▉        | 246/1261 [06:22<25:49,  1.53s/it][A
     20%|█▉        | 247/1261 [06:24<26:03,  1.54s/it][A
     20%|█▉        | 248/1261 [06:25<25:53,  1.53s/it][A
     20%|█▉        | 249/1261 [06:27<26:01,  1.54s/it][A
     20%|█▉        | 250/1261 [06:28<25:59,  1.54s/it][A
     20%|█▉        | 251/1261 [06:30<25:40,  1.53s/it][A
     20%|█▉        | 252/1261 [06:31<25:42,  1.53s/it][A
     20%|██        | 253/1261 [06:33<25:59,  1.55s/it][A
     20%|██        | 254/1261 [06:34<25:57,  1.55s/it][A
     20%|██        | 255/1261 [06:36<25:58,  1.55s/it][A
     20%|██        | 256/1261 [06:38<25:53,  1.55s/it][A
     20%|██        | 257/1261 [06:39<26:03,  1.56s/it][A
     20%|██        | 258/1261 [06:41<26:04,  1.56s/it][A
     21%|██        | 259/1261 [06:42<25:44,  1.54s/it][A
     21%|██        | 260/1261 [06:44<25:30,  1.53s/it][A
     21%|██        | 261/1261 [06:45<25:11,  1.51s/it][A
     21%|██        | 262/1261 [06:47<25:09,  1.51s/it][A
     21%|██        | 263/1261 [06:48<25:16,  1.52s/it][A
     21%|██        | 264/1261 [06:50<25:21,  1.53s/it][A
     21%|██        | 265/1261 [06:51<25:44,  1.55s/it][A
     21%|██        | 266/1261 [06:53<26:01,  1.57s/it][A
     21%|██        | 267/1261 [06:55<26:06,  1.58s/it][A
     21%|██▏       | 268/1261 [06:56<25:57,  1.57s/it][A
     21%|██▏       | 269/1261 [06:58<25:39,  1.55s/it][A
     21%|██▏       | 270/1261 [06:59<25:31,  1.55s/it][A
     21%|██▏       | 271/1261 [07:01<25:23,  1.54s/it][A
     22%|██▏       | 272/1261 [07:02<25:20,  1.54s/it][A
     22%|██▏       | 273/1261 [07:04<25:16,  1.53s/it][A
     22%|██▏       | 274/1261 [07:05<25:35,  1.56s/it][A
     22%|██▏       | 275/1261 [07:07<25:14,  1.54s/it][A
     22%|██▏       | 276/1261 [07:08<25:17,  1.54s/it][A
     22%|██▏       | 277/1261 [07:10<25:32,  1.56s/it][A
     22%|██▏       | 278/1261 [07:12<25:37,  1.56s/it][A
     22%|██▏       | 279/1261 [07:13<25:31,  1.56s/it][A
     22%|██▏       | 280/1261 [07:15<25:24,  1.55s/it][A
     22%|██▏       | 281/1261 [07:16<25:13,  1.54s/it][A
     22%|██▏       | 282/1261 [07:18<25:02,  1.53s/it][A
     22%|██▏       | 283/1261 [07:19<24:56,  1.53s/it][A
     23%|██▎       | 284/1261 [07:21<25:01,  1.54s/it][A
     23%|██▎       | 285/1261 [07:22<24:40,  1.52s/it][A
     23%|██▎       | 286/1261 [07:24<24:48,  1.53s/it][A
     23%|██▎       | 287/1261 [07:25<25:01,  1.54s/it][A
     23%|██▎       | 288/1261 [07:27<25:17,  1.56s/it][A
     23%|██▎       | 289/1261 [07:29<25:12,  1.56s/it][A
     23%|██▎       | 290/1261 [07:30<25:12,  1.56s/it][A
     23%|██▎       | 291/1261 [07:32<25:04,  1.55s/it][A
     23%|██▎       | 292/1261 [07:33<24:54,  1.54s/it][A
     23%|██▎       | 293/1261 [07:35<24:49,  1.54s/it][A
     23%|██▎       | 294/1261 [07:36<24:57,  1.55s/it][A
     23%|██▎       | 295/1261 [07:38<25:13,  1.57s/it][A
     23%|██▎       | 296/1261 [07:40<25:28,  1.58s/it][A
     24%|██▎       | 297/1261 [07:41<25:27,  1.58s/it][A
     24%|██▎       | 298/1261 [07:43<25:40,  1.60s/it][A
     24%|██▎       | 299/1261 [07:44<25:35,  1.60s/it][A
     24%|██▍       | 300/1261 [07:46<26:41,  1.67s/it][A
     24%|██▍       | 301/1261 [07:48<26:26,  1.65s/it][A
     24%|██▍       | 302/1261 [07:49<26:45,  1.67s/it][A
     24%|██▍       | 303/1261 [07:51<26:29,  1.66s/it][A
     24%|██▍       | 304/1261 [07:53<26:21,  1.65s/it][A
     24%|██▍       | 305/1261 [07:54<25:50,  1.62s/it][A
     24%|██▍       | 306/1261 [07:56<25:34,  1.61s/it][A
     24%|██▍       | 307/1261 [07:57<25:27,  1.60s/it][A
     24%|██▍       | 308/1261 [07:59<25:05,  1.58s/it][A
     25%|██▍       | 309/1261 [08:01<24:52,  1.57s/it][A
     25%|██▍       | 310/1261 [08:02<25:07,  1.59s/it][A
     25%|██▍       | 311/1261 [08:04<25:04,  1.58s/it][A
     25%|██▍       | 312/1261 [08:05<24:52,  1.57s/it][A
     25%|██▍       | 313/1261 [08:07<24:45,  1.57s/it][A
     25%|██▍       | 314/1261 [08:08<24:51,  1.58s/it][A
     25%|██▍       | 315/1261 [08:10<24:50,  1.58s/it][A
     25%|██▌       | 316/1261 [08:12<24:35,  1.56s/it][A
     25%|██▌       | 317/1261 [08:13<24:26,  1.55s/it][A
     25%|██▌       | 318/1261 [08:15<24:23,  1.55s/it][A
     25%|██▌       | 319/1261 [08:16<24:21,  1.55s/it][A
     25%|██▌       | 320/1261 [08:18<24:20,  1.55s/it][A
     25%|██▌       | 321/1261 [08:19<24:35,  1.57s/it][A
     26%|██▌       | 322/1261 [08:21<24:35,  1.57s/it][A
     26%|██▌       | 323/1261 [08:22<24:27,  1.56s/it][A
     26%|██▌       | 324/1261 [08:24<24:26,  1.57s/it][A
     26%|██▌       | 325/1261 [08:26<24:22,  1.56s/it][A
     26%|██▌       | 326/1261 [08:27<24:18,  1.56s/it][A
     26%|██▌       | 327/1261 [08:29<24:17,  1.56s/it][A
     26%|██▌       | 328/1261 [08:30<24:18,  1.56s/it][A
     26%|██▌       | 329/1261 [08:32<24:05,  1.55s/it][A
     26%|██▌       | 330/1261 [08:33<23:55,  1.54s/it][A
     26%|██▌       | 331/1261 [08:35<23:56,  1.54s/it][A
     26%|██▋       | 332/1261 [08:36<23:49,  1.54s/it][A
     26%|██▋       | 333/1261 [08:38<23:56,  1.55s/it][A
     26%|██▋       | 334/1261 [08:39<23:51,  1.54s/it][A
     27%|██▋       | 335/1261 [08:41<23:45,  1.54s/it][A
     27%|██▋       | 336/1261 [08:43<23:44,  1.54s/it][A
     27%|██▋       | 337/1261 [08:44<23:48,  1.55s/it][A
     27%|██▋       | 338/1261 [08:46<23:55,  1.56s/it][A
     27%|██▋       | 339/1261 [08:47<24:19,  1.58s/it][A
     27%|██▋       | 340/1261 [08:49<24:05,  1.57s/it][A
     27%|██▋       | 341/1261 [08:50<24:08,  1.57s/it][A
     27%|██▋       | 342/1261 [08:52<23:53,  1.56s/it][A
     27%|██▋       | 343/1261 [08:54<23:42,  1.55s/it][A
     27%|██▋       | 344/1261 [08:55<23:35,  1.54s/it][A
     27%|██▋       | 345/1261 [08:57<23:46,  1.56s/it][A
     27%|██▋       | 346/1261 [08:58<23:31,  1.54s/it][A
     28%|██▊       | 347/1261 [09:00<23:41,  1.56s/it][A
     28%|██▊       | 348/1261 [09:01<23:42,  1.56s/it][A
     28%|██▊       | 349/1261 [09:03<23:37,  1.55s/it][A
     28%|██▊       | 350/1261 [09:04<23:29,  1.55s/it][A
     28%|██▊       | 351/1261 [09:06<23:36,  1.56s/it][A
     28%|██▊       | 352/1261 [09:07<23:30,  1.55s/it][A
     28%|██▊       | 353/1261 [09:09<23:32,  1.56s/it][A
     28%|██▊       | 354/1261 [09:11<23:24,  1.55s/it][A
     28%|██▊       | 355/1261 [09:12<23:24,  1.55s/it][A
     28%|██▊       | 356/1261 [09:14<23:25,  1.55s/it][A
     28%|██▊       | 357/1261 [09:15<24:01,  1.60s/it][A
     28%|██▊       | 358/1261 [09:17<23:45,  1.58s/it][A
     28%|██▊       | 359/1261 [09:19<23:44,  1.58s/it][A
     29%|██▊       | 360/1261 [09:20<23:36,  1.57s/it][A
     29%|██▊       | 361/1261 [09:22<23:32,  1.57s/it][A
     29%|██▊       | 362/1261 [09:23<23:33,  1.57s/it][A
     29%|██▉       | 363/1261 [09:25<23:48,  1.59s/it][A
     29%|██▉       | 364/1261 [09:26<23:34,  1.58s/it][A
     29%|██▉       | 365/1261 [09:28<23:21,  1.56s/it][A
     29%|██▉       | 366/1261 [09:29<23:11,  1.55s/it][A
     29%|██▉       | 367/1261 [09:31<23:15,  1.56s/it][A
     29%|██▉       | 368/1261 [09:33<23:13,  1.56s/it][A
     29%|██▉       | 369/1261 [09:34<23:12,  1.56s/it][A
     29%|██▉       | 370/1261 [09:36<23:07,  1.56s/it][A
     29%|██▉       | 371/1261 [09:37<23:11,  1.56s/it][A
     30%|██▉       | 372/1261 [09:39<23:15,  1.57s/it][A
     30%|██▉       | 373/1261 [09:40<23:17,  1.57s/it][A
     30%|██▉       | 374/1261 [09:42<23:13,  1.57s/it][A
     30%|██▉       | 375/1261 [09:44<23:08,  1.57s/it][A
     30%|██▉       | 376/1261 [09:45<23:08,  1.57s/it][A
     30%|██▉       | 377/1261 [09:47<24:17,  1.65s/it][A
     30%|██▉       | 378/1261 [09:49<25:24,  1.73s/it][A
     30%|███       | 379/1261 [09:50<24:50,  1.69s/it][A
     30%|███       | 380/1261 [09:52<24:34,  1.67s/it][A
     30%|███       | 381/1261 [09:54<24:13,  1.65s/it][A
     30%|███       | 382/1261 [09:55<23:44,  1.62s/it][A
     30%|███       | 383/1261 [09:57<23:30,  1.61s/it][A
     30%|███       | 384/1261 [09:58<23:09,  1.58s/it][A
     31%|███       | 385/1261 [10:00<23:09,  1.59s/it][A
     31%|███       | 386/1261 [10:02<23:06,  1.58s/it][A
     31%|███       | 387/1261 [10:03<23:09,  1.59s/it][A
     31%|███       | 388/1261 [10:05<23:04,  1.59s/it][A
     31%|███       | 389/1261 [10:06<22:54,  1.58s/it][A
     31%|███       | 390/1261 [10:08<22:48,  1.57s/it][A
     31%|███       | 391/1261 [10:09<22:48,  1.57s/it][A
     31%|███       | 392/1261 [10:11<23:00,  1.59s/it][A
     31%|███       | 393/1261 [10:13<22:48,  1.58s/it][A
     31%|███       | 394/1261 [10:14<22:37,  1.57s/it][A
     31%|███▏      | 395/1261 [10:16<22:35,  1.57s/it][A
     31%|███▏      | 396/1261 [10:17<22:36,  1.57s/it][A
     31%|███▏      | 397/1261 [10:19<22:44,  1.58s/it][A
     32%|███▏      | 398/1261 [10:20<22:38,  1.57s/it][A
     32%|███▏      | 399/1261 [10:22<22:46,  1.58s/it][A
     32%|███▏      | 400/1261 [10:24<22:47,  1.59s/it][A
     32%|███▏      | 401/1261 [10:25<22:51,  1.59s/it][A
     32%|███▏      | 402/1261 [10:27<23:18,  1.63s/it][A
     32%|███▏      | 403/1261 [10:29<23:11,  1.62s/it][A
     32%|███▏      | 404/1261 [10:30<22:58,  1.61s/it][A
     32%|███▏      | 405/1261 [10:32<22:53,  1.60s/it][A
     32%|███▏      | 406/1261 [10:33<22:44,  1.60s/it][A
     32%|███▏      | 407/1261 [10:35<22:33,  1.59s/it][A
     32%|███▏      | 408/1261 [10:36<22:33,  1.59s/it][A
     32%|███▏      | 409/1261 [10:38<22:54,  1.61s/it][A
     33%|███▎      | 410/1261 [10:40<22:55,  1.62s/it][A
     33%|███▎      | 411/1261 [10:41<22:56,  1.62s/it][A
     33%|███▎      | 412/1261 [10:43<22:31,  1.59s/it][A
     33%|███▎      | 413/1261 [10:44<22:12,  1.57s/it][A
     33%|███▎      | 414/1261 [10:46<21:59,  1.56s/it][A
     33%|███▎      | 415/1261 [10:48<21:56,  1.56s/it][A
     33%|███▎      | 416/1261 [10:49<21:59,  1.56s/it][A
     33%|███▎      | 417/1261 [10:51<21:54,  1.56s/it][A
     33%|███▎      | 418/1261 [10:52<22:01,  1.57s/it][A
     33%|███▎      | 419/1261 [10:54<21:57,  1.57s/it][A
     33%|███▎      | 420/1261 [10:55<21:54,  1.56s/it][A
     33%|███▎      | 421/1261 [10:57<21:50,  1.56s/it][A
     33%|███▎      | 422/1261 [10:59<22:05,  1.58s/it][A
     34%|███▎      | 423/1261 [11:00<21:59,  1.57s/it][A
     34%|███▎      | 424/1261 [11:02<21:50,  1.57s/it][A
     34%|███▎      | 425/1261 [11:03<21:40,  1.56s/it][A
     34%|███▍      | 426/1261 [11:05<21:41,  1.56s/it][A
     34%|███▍      | 427/1261 [11:06<21:39,  1.56s/it][A
     34%|███▍      | 428/1261 [11:08<21:39,  1.56s/it][A
     34%|███▍      | 429/1261 [11:09<21:27,  1.55s/it][A
     34%|███▍      | 430/1261 [11:11<21:24,  1.55s/it][A
     34%|███▍      | 431/1261 [11:12<21:16,  1.54s/it][A
     34%|███▍      | 432/1261 [11:14<21:15,  1.54s/it][A
     34%|███▍      | 433/1261 [11:16<21:31,  1.56s/it][A
     34%|███▍      | 434/1261 [11:17<21:32,  1.56s/it][A
     34%|███▍      | 435/1261 [11:19<21:39,  1.57s/it][A
     35%|███▍      | 436/1261 [11:20<21:51,  1.59s/it][A
     35%|███▍      | 437/1261 [11:22<21:38,  1.58s/it][A
     35%|███▍      | 438/1261 [11:23<21:27,  1.56s/it][A
     35%|███▍      | 439/1261 [11:25<21:16,  1.55s/it][A
     35%|███▍      | 440/1261 [11:27<21:15,  1.55s/it][A
     35%|███▍      | 441/1261 [11:28<21:14,  1.55s/it][A
     35%|███▌      | 442/1261 [11:30<21:14,  1.56s/it][A
     35%|███▌      | 443/1261 [11:31<21:12,  1.56s/it][A
     35%|███▌      | 444/1261 [11:33<21:16,  1.56s/it][A
     35%|███▌      | 445/1261 [11:34<21:19,  1.57s/it][A
     35%|███▌      | 446/1261 [11:36<21:28,  1.58s/it][A
     35%|███▌      | 447/1261 [11:38<21:29,  1.58s/it][A
     36%|███▌      | 448/1261 [11:39<21:24,  1.58s/it][A
     36%|███▌      | 449/1261 [11:41<21:19,  1.58s/it][A
     36%|███▌      | 450/1261 [11:42<21:11,  1.57s/it][A
     36%|███▌      | 451/1261 [11:44<21:17,  1.58s/it][A
     36%|███▌      | 452/1261 [11:45<21:11,  1.57s/it][A
     36%|███▌      | 453/1261 [11:47<22:11,  1.65s/it][A
     36%|███▌      | 454/1261 [11:49<22:03,  1.64s/it][A
     36%|███▌      | 455/1261 [11:51<22:02,  1.64s/it][A
     36%|███▌      | 456/1261 [11:52<21:57,  1.64s/it][A
     36%|███▌      | 457/1261 [11:54<21:55,  1.64s/it][A
     36%|███▋      | 458/1261 [11:55<21:44,  1.62s/it][A
     36%|███▋      | 459/1261 [11:57<21:27,  1.61s/it][A
     36%|███▋      | 460/1261 [11:58<21:12,  1.59s/it][A
     37%|███▋      | 461/1261 [12:00<21:04,  1.58s/it][A
     37%|███▋      | 462/1261 [12:02<20:50,  1.56s/it][A
     37%|███▋      | 463/1261 [12:03<20:53,  1.57s/it][A
     37%|███▋      | 464/1261 [12:05<20:40,  1.56s/it][A
     37%|███▋      | 465/1261 [12:06<20:36,  1.55s/it][A
     37%|███▋      | 466/1261 [12:08<20:32,  1.55s/it][A
     37%|███▋      | 467/1261 [12:09<20:32,  1.55s/it][A
     37%|███▋      | 468/1261 [12:11<20:35,  1.56s/it][A
     37%|███▋      | 469/1261 [12:13<20:45,  1.57s/it][A
     37%|███▋      | 470/1261 [12:14<20:41,  1.57s/it][A
     37%|███▋      | 471/1261 [12:16<20:41,  1.57s/it][A
     37%|███▋      | 472/1261 [12:17<20:39,  1.57s/it][A
     38%|███▊      | 473/1261 [12:19<20:28,  1.56s/it][A
     38%|███▊      | 474/1261 [12:20<20:24,  1.56s/it][A
     38%|███▊      | 475/1261 [12:22<20:35,  1.57s/it][A
     38%|███▊      | 476/1261 [12:23<20:23,  1.56s/it][A
     38%|███▊      | 477/1261 [12:25<20:19,  1.55s/it][A
     38%|███▊      | 478/1261 [12:27<20:18,  1.56s/it][A
     38%|███▊      | 479/1261 [12:28<20:28,  1.57s/it][A
     38%|███▊      | 480/1261 [12:30<20:28,  1.57s/it][A
     38%|███▊      | 481/1261 [12:31<20:28,  1.57s/it][A
     38%|███▊      | 482/1261 [12:33<20:35,  1.59s/it][A
     38%|███▊      | 483/1261 [12:34<20:30,  1.58s/it][A
     38%|███▊      | 484/1261 [12:36<20:26,  1.58s/it][A
     38%|███▊      | 485/1261 [12:38<20:21,  1.57s/it][A
     39%|███▊      | 486/1261 [12:39<20:20,  1.57s/it][A
     39%|███▊      | 487/1261 [12:41<20:15,  1.57s/it][A
     39%|███▊      | 488/1261 [12:42<20:12,  1.57s/it][A
     39%|███▉      | 489/1261 [12:44<20:22,  1.58s/it][A
     39%|███▉      | 490/1261 [12:46<20:25,  1.59s/it][A
     39%|███▉      | 491/1261 [12:47<20:28,  1.60s/it][A
     39%|███▉      | 492/1261 [12:49<20:24,  1.59s/it][A
     39%|███▉      | 493/1261 [12:50<20:18,  1.59s/it][A
     39%|███▉      | 494/1261 [12:52<20:20,  1.59s/it][A
     39%|███▉      | 495/1261 [12:54<20:23,  1.60s/it][A
     39%|███▉      | 496/1261 [12:55<20:16,  1.59s/it][A
     39%|███▉      | 497/1261 [12:57<20:27,  1.61s/it][A
     39%|███▉      | 498/1261 [12:58<20:17,  1.60s/it][A
     40%|███▉      | 499/1261 [13:00<20:19,  1.60s/it][A
     40%|███▉      | 500/1261 [13:01<20:11,  1.59s/it][A
     40%|███▉      | 501/1261 [13:03<20:09,  1.59s/it][A
     40%|███▉      | 502/1261 [13:05<20:03,  1.59s/it][A
     40%|███▉      | 503/1261 [13:06<20:12,  1.60s/it][A
     40%|███▉      | 504/1261 [13:08<20:02,  1.59s/it][A
     40%|████      | 505/1261 [13:09<20:02,  1.59s/it][A
     40%|████      | 506/1261 [13:11<20:01,  1.59s/it][A
     40%|████      | 507/1261 [13:13<20:03,  1.60s/it][A
     40%|████      | 508/1261 [13:14<19:58,  1.59s/it][A
     40%|████      | 509/1261 [13:16<19:52,  1.59s/it][A
     40%|████      | 510/1261 [13:17<19:42,  1.57s/it][A
     41%|████      | 511/1261 [13:19<19:42,  1.58s/it][A
     41%|████      | 512/1261 [13:21<19:46,  1.58s/it][A
     41%|████      | 513/1261 [13:22<19:47,  1.59s/it][A
     41%|████      | 514/1261 [13:24<19:40,  1.58s/it][A
     41%|████      | 515/1261 [13:25<19:52,  1.60s/it][A
     41%|████      | 516/1261 [13:27<19:54,  1.60s/it][A
     41%|████      | 517/1261 [13:29<20:03,  1.62s/it][A
     41%|████      | 518/1261 [13:30<19:59,  1.61s/it][A
     41%|████      | 519/1261 [13:32<19:47,  1.60s/it][A
     41%|████      | 520/1261 [13:33<19:44,  1.60s/it][A
     41%|████▏     | 521/1261 [13:35<19:36,  1.59s/it][A
     41%|████▏     | 522/1261 [13:37<19:32,  1.59s/it][A
     41%|████▏     | 523/1261 [13:38<19:35,  1.59s/it][A
     42%|████▏     | 524/1261 [13:40<19:27,  1.58s/it][A
     42%|████▏     | 525/1261 [13:41<19:20,  1.58s/it][A
     42%|████▏     | 526/1261 [13:43<19:02,  1.55s/it][A
     42%|████▏     | 527/1261 [13:44<19:08,  1.57s/it][A
     42%|████▏     | 528/1261 [13:46<19:25,  1.59s/it][A
     42%|████▏     | 529/1261 [13:48<19:46,  1.62s/it][A
     42%|████▏     | 530/1261 [13:49<19:25,  1.59s/it][A
     42%|████▏     | 531/1261 [13:51<19:30,  1.60s/it][A
     42%|████▏     | 532/1261 [13:52<19:14,  1.58s/it][A
     42%|████▏     | 533/1261 [13:54<18:59,  1.56s/it][A
     42%|████▏     | 534/1261 [13:55<18:48,  1.55s/it][A
     42%|████▏     | 535/1261 [13:57<18:34,  1.53s/it][A
     43%|████▎     | 536/1261 [13:58<18:22,  1.52s/it][A
     43%|████▎     | 537/1261 [14:00<18:09,  1.51s/it][A
     43%|████▎     | 538/1261 [14:01<18:02,  1.50s/it][A
     43%|████▎     | 539/1261 [14:03<17:59,  1.50s/it][A
     43%|████▎     | 540/1261 [14:04<18:03,  1.50s/it][A
     43%|████▎     | 541/1261 [14:06<18:14,  1.52s/it][A
     43%|████▎     | 542/1261 [14:07<18:14,  1.52s/it][A
     43%|████▎     | 543/1261 [14:09<18:24,  1.54s/it][A
     43%|████▎     | 544/1261 [14:11<18:25,  1.54s/it][A
     43%|████▎     | 545/1261 [14:12<18:42,  1.57s/it][A
     43%|████▎     | 546/1261 [14:14<18:53,  1.59s/it][A
     43%|████▎     | 547/1261 [14:15<19:08,  1.61s/it][A
     43%|████▎     | 548/1261 [14:17<19:21,  1.63s/it][A
     44%|████▎     | 549/1261 [14:19<19:38,  1.66s/it][A
     44%|████▎     | 550/1261 [14:21<20:01,  1.69s/it][A
     44%|████▎     | 551/1261 [14:22<20:21,  1.72s/it][A
     44%|████▍     | 552/1261 [14:24<20:40,  1.75s/it][A
     44%|████▍     | 553/1261 [14:26<20:55,  1.77s/it][A
     44%|████▍     | 554/1261 [14:28<20:50,  1.77s/it][A
     44%|████▍     | 555/1261 [14:30<20:40,  1.76s/it][A
     44%|████▍     | 556/1261 [14:31<20:27,  1.74s/it][A
     44%|████▍     | 557/1261 [14:33<20:11,  1.72s/it][A
     44%|████▍     | 558/1261 [14:35<20:01,  1.71s/it][A
     44%|████▍     | 559/1261 [14:36<19:46,  1.69s/it][A
     44%|████▍     | 560/1261 [14:38<19:39,  1.68s/it][A
     44%|████▍     | 561/1261 [14:40<19:35,  1.68s/it][A
     45%|████▍     | 562/1261 [14:41<19:30,  1.67s/it][A
     45%|████▍     | 563/1261 [14:43<19:25,  1.67s/it][A
     45%|████▍     | 564/1261 [14:45<19:39,  1.69s/it][A
     45%|████▍     | 565/1261 [14:46<19:29,  1.68s/it][A
     45%|████▍     | 566/1261 [14:48<19:17,  1.67s/it][A
     45%|████▍     | 567/1261 [14:50<19:10,  1.66s/it][A
     45%|████▌     | 568/1261 [14:51<19:02,  1.65s/it][A
     45%|████▌     | 569/1261 [14:53<18:54,  1.64s/it][A
     45%|████▌     | 570/1261 [14:54<18:51,  1.64s/it][A
     45%|████▌     | 571/1261 [14:56<18:44,  1.63s/it][A
     45%|████▌     | 572/1261 [14:58<18:43,  1.63s/it][A
     45%|████▌     | 573/1261 [14:59<18:52,  1.65s/it][A
     46%|████▌     | 574/1261 [15:01<18:47,  1.64s/it][A
     46%|████▌     | 575/1261 [15:03<18:40,  1.63s/it][A
     46%|████▌     | 576/1261 [15:04<18:33,  1.63s/it][A
     46%|████▌     | 577/1261 [15:06<18:32,  1.63s/it][A
     46%|████▌     | 578/1261 [15:08<18:29,  1.62s/it][A
     46%|████▌     | 579/1261 [15:09<18:15,  1.61s/it][A
     46%|████▌     | 580/1261 [15:11<18:11,  1.60s/it][A
     46%|████▌     | 581/1261 [15:12<18:10,  1.60s/it][A
     46%|████▌     | 582/1261 [15:14<18:05,  1.60s/it][A
     46%|████▌     | 583/1261 [15:16<18:20,  1.62s/it][A
     46%|████▋     | 584/1261 [15:17<18:15,  1.62s/it][A
     46%|████▋     | 585/1261 [15:19<18:19,  1.63s/it][A
     46%|████▋     | 586/1261 [15:21<18:35,  1.65s/it][A
     47%|████▋     | 587/1261 [15:22<18:36,  1.66s/it][A
     47%|████▋     | 588/1261 [15:24<18:23,  1.64s/it][A
     47%|████▋     | 589/1261 [15:25<18:09,  1.62s/it][A
     47%|████▋     | 590/1261 [15:27<18:10,  1.63s/it][A
     47%|████▋     | 591/1261 [15:29<18:16,  1.64s/it][A
     47%|████▋     | 592/1261 [15:30<18:09,  1.63s/it][A
     47%|████▋     | 593/1261 [15:32<18:08,  1.63s/it][A
     47%|████▋     | 594/1261 [15:33<18:01,  1.62s/it][A
     47%|████▋     | 595/1261 [15:35<18:03,  1.63s/it][A
     47%|████▋     | 596/1261 [15:37<18:06,  1.63s/it][A
     47%|████▋     | 597/1261 [15:38<18:05,  1.64s/it][A
     47%|████▋     | 598/1261 [15:40<17:59,  1.63s/it][A
     48%|████▊     | 599/1261 [15:42<17:52,  1.62s/it][A
     48%|████▊     | 600/1261 [15:43<17:55,  1.63s/it][A
     48%|████▊     | 601/1261 [15:45<17:53,  1.63s/it][A
     48%|████▊     | 602/1261 [15:47<18:04,  1.65s/it][A
     48%|████▊     | 603/1261 [15:48<18:30,  1.69s/it][A
     48%|████▊     | 604/1261 [15:50<18:30,  1.69s/it][A
     48%|████▊     | 605/1261 [15:52<18:11,  1.66s/it][A
     48%|████▊     | 606/1261 [15:53<18:06,  1.66s/it][A
     48%|████▊     | 607/1261 [15:55<18:00,  1.65s/it][A
     48%|████▊     | 608/1261 [15:57<17:49,  1.64s/it][A
     48%|████▊     | 609/1261 [15:58<17:44,  1.63s/it][A
     48%|████▊     | 610/1261 [16:00<17:36,  1.62s/it][A
     48%|████▊     | 611/1261 [16:01<17:16,  1.59s/it][A
     49%|████▊     | 612/1261 [16:03<16:55,  1.56s/it][A
     49%|████▊     | 613/1261 [16:04<16:39,  1.54s/it][A
     49%|████▊     | 614/1261 [16:06<16:35,  1.54s/it][A
     49%|████▉     | 615/1261 [16:07<16:36,  1.54s/it][A
     49%|████▉     | 616/1261 [16:09<16:49,  1.57s/it][A
     49%|████▉     | 617/1261 [16:11<16:53,  1.57s/it][A
     49%|████▉     | 618/1261 [16:12<16:36,  1.55s/it][A
     49%|████▉     | 619/1261 [16:14<16:19,  1.53s/it][A
     49%|████▉     | 620/1261 [16:15<16:13,  1.52s/it][A
     49%|████▉     | 621/1261 [16:17<16:00,  1.50s/it][A
     49%|████▉     | 622/1261 [16:18<15:50,  1.49s/it][A
     49%|████▉     | 623/1261 [16:19<15:46,  1.48s/it][A
     49%|████▉     | 624/1261 [16:21<15:51,  1.49s/it][A
     50%|████▉     | 625/1261 [16:23<16:05,  1.52s/it][A
     50%|████▉     | 626/1261 [16:24<16:09,  1.53s/it][A
     50%|████▉     | 627/1261 [16:26<16:03,  1.52s/it][A
     50%|████▉     | 628/1261 [16:27<15:57,  1.51s/it][A
     50%|████▉     | 629/1261 [16:29<15:46,  1.50s/it][A
     50%|████▉     | 630/1261 [16:30<15:51,  1.51s/it][A
     50%|█████     | 631/1261 [16:32<15:50,  1.51s/it][A
     50%|█████     | 632/1261 [16:33<15:57,  1.52s/it][A
     50%|█████     | 633/1261 [16:35<15:54,  1.52s/it][A
     50%|█████     | 634/1261 [16:36<15:57,  1.53s/it][A
     50%|█████     | 635/1261 [16:38<15:48,  1.52s/it][A
     50%|█████     | 636/1261 [16:39<15:46,  1.51s/it][A
     51%|█████     | 637/1261 [16:41<15:50,  1.52s/it][A
     51%|█████     | 638/1261 [16:42<15:50,  1.53s/it][A
     51%|█████     | 639/1261 [16:44<16:00,  1.54s/it][A
     51%|█████     | 640/1261 [16:45<15:56,  1.54s/it][A
     51%|█████     | 641/1261 [16:47<15:58,  1.55s/it][A
     51%|█████     | 642/1261 [16:49<15:57,  1.55s/it][A
     51%|█████     | 643/1261 [16:50<15:57,  1.55s/it][A
     51%|█████     | 644/1261 [16:52<15:57,  1.55s/it][A
     51%|█████     | 645/1261 [16:53<15:53,  1.55s/it][A
     51%|█████     | 646/1261 [16:55<15:51,  1.55s/it][A
     51%|█████▏    | 647/1261 [16:56<15:48,  1.55s/it][A
     51%|█████▏    | 648/1261 [16:58<15:59,  1.57s/it][A
     51%|█████▏    | 649/1261 [16:59<15:53,  1.56s/it][A
     52%|█████▏    | 650/1261 [17:01<15:54,  1.56s/it][A
     52%|█████▏    | 651/1261 [17:03<16:02,  1.58s/it][A
     52%|█████▏    | 652/1261 [17:04<16:03,  1.58s/it][A
     52%|█████▏    | 653/1261 [17:06<15:52,  1.57s/it][A
     52%|█████▏    | 654/1261 [17:07<15:44,  1.56s/it][A
     52%|█████▏    | 655/1261 [17:09<15:42,  1.56s/it][A
     52%|█████▏    | 656/1261 [17:10<15:37,  1.55s/it][A
     52%|█████▏    | 657/1261 [17:12<15:37,  1.55s/it][A
     52%|█████▏    | 658/1261 [17:13<15:35,  1.55s/it][A
     52%|█████▏    | 659/1261 [17:15<15:28,  1.54s/it][A
     52%|█████▏    | 660/1261 [17:16<15:25,  1.54s/it][A
     52%|█████▏    | 661/1261 [17:18<15:35,  1.56s/it][A
     52%|█████▏    | 662/1261 [17:20<15:41,  1.57s/it][A
     53%|█████▎    | 663/1261 [17:21<15:40,  1.57s/it][A
     53%|█████▎    | 664/1261 [17:23<15:40,  1.58s/it][A
     53%|█████▎    | 665/1261 [17:24<15:41,  1.58s/it][A
     53%|█████▎    | 666/1261 [17:26<15:32,  1.57s/it][A
     53%|█████▎    | 667/1261 [17:28<15:26,  1.56s/it][A
     53%|█████▎    | 668/1261 [17:29<15:22,  1.56s/it][A
     53%|█████▎    | 669/1261 [17:31<15:15,  1.55s/it][A
     53%|█████▎    | 670/1261 [17:32<15:11,  1.54s/it][A
     53%|█████▎    | 671/1261 [17:34<15:09,  1.54s/it][A
     53%|█████▎    | 672/1261 [17:35<15:06,  1.54s/it][A
     53%|█████▎    | 673/1261 [17:37<15:12,  1.55s/it][A
     53%|█████▎    | 674/1261 [17:38<15:12,  1.56s/it][A
     54%|█████▎    | 675/1261 [17:40<15:13,  1.56s/it][A
     54%|█████▎    | 676/1261 [17:42<15:17,  1.57s/it][A
     54%|█████▎    | 677/1261 [17:43<15:21,  1.58s/it][A
     54%|█████▍    | 678/1261 [17:45<15:23,  1.58s/it][A
     54%|█████▍    | 679/1261 [17:46<15:20,  1.58s/it][A
     54%|█████▍    | 680/1261 [17:48<15:30,  1.60s/it][A
     54%|█████▍    | 681/1261 [17:49<15:22,  1.59s/it][A
     54%|█████▍    | 682/1261 [17:51<15:24,  1.60s/it][A
     54%|█████▍    | 683/1261 [17:53<15:20,  1.59s/it][A
     54%|█████▍    | 684/1261 [17:54<15:14,  1.59s/it][A
     54%|█████▍    | 685/1261 [17:56<15:12,  1.58s/it][A
     54%|█████▍    | 686/1261 [17:57<15:15,  1.59s/it][A
     54%|█████▍    | 687/1261 [17:59<15:17,  1.60s/it][A
     55%|█████▍    | 688/1261 [18:01<15:19,  1.60s/it][A
     55%|█████▍    | 689/1261 [18:02<15:12,  1.59s/it][A
     55%|█████▍    | 690/1261 [18:04<15:03,  1.58s/it][A
     55%|█████▍    | 691/1261 [18:05<15:00,  1.58s/it][A
     55%|█████▍    | 692/1261 [18:07<15:05,  1.59s/it][A
     55%|█████▍    | 693/1261 [18:09<15:00,  1.59s/it][A
     55%|█████▌    | 694/1261 [18:10<15:19,  1.62s/it][A
     55%|█████▌    | 695/1261 [18:12<15:17,  1.62s/it][A
     55%|█████▌    | 696/1261 [18:13<15:00,  1.59s/it][A
     55%|█████▌    | 697/1261 [18:15<14:52,  1.58s/it][A
     55%|█████▌    | 698/1261 [18:17<14:43,  1.57s/it][A
     55%|█████▌    | 699/1261 [18:18<14:42,  1.57s/it][A
     56%|█████▌    | 700/1261 [18:20<14:41,  1.57s/it][A
     56%|█████▌    | 701/1261 [18:21<14:35,  1.56s/it][A
     56%|█████▌    | 702/1261 [18:23<14:35,  1.57s/it][A
     56%|█████▌    | 703/1261 [18:24<14:27,  1.55s/it][A
     56%|█████▌    | 704/1261 [18:26<14:23,  1.55s/it][A
     56%|█████▌    | 705/1261 [18:27<14:31,  1.57s/it][A
     56%|█████▌    | 706/1261 [18:29<14:23,  1.56s/it][A
     56%|█████▌    | 707/1261 [18:31<14:21,  1.56s/it][A
     56%|█████▌    | 708/1261 [18:32<14:20,  1.56s/it][A
     56%|█████▌    | 709/1261 [18:34<14:26,  1.57s/it][A
     56%|█████▋    | 710/1261 [18:35<14:19,  1.56s/it][A
     56%|█████▋    | 711/1261 [18:37<14:13,  1.55s/it][A
     56%|█████▋    | 712/1261 [18:38<14:14,  1.56s/it][A
     57%|█████▋    | 713/1261 [18:40<14:13,  1.56s/it][A
     57%|█████▋    | 714/1261 [18:41<14:16,  1.57s/it][A
     57%|█████▋    | 715/1261 [18:43<14:10,  1.56s/it][A
     57%|█████▋    | 716/1261 [18:45<14:24,  1.59s/it][A
     57%|█████▋    | 717/1261 [18:46<14:15,  1.57s/it][A
     57%|█████▋    | 718/1261 [18:48<14:09,  1.56s/it][A
     57%|█████▋    | 719/1261 [18:49<14:09,  1.57s/it][A
     57%|█████▋    | 720/1261 [18:51<14:11,  1.57s/it][A
     57%|█████▋    | 721/1261 [18:52<14:03,  1.56s/it][A
     57%|█████▋    | 722/1261 [18:54<14:03,  1.56s/it][A
     57%|█████▋    | 723/1261 [18:56<14:01,  1.56s/it][A
     57%|█████▋    | 724/1261 [18:57<14:00,  1.57s/it][A
     57%|█████▋    | 725/1261 [18:59<13:56,  1.56s/it][A
     58%|█████▊    | 726/1261 [19:00<14:01,  1.57s/it][A
     58%|█████▊    | 727/1261 [19:02<13:54,  1.56s/it][A
     58%|█████▊    | 728/1261 [19:03<13:49,  1.56s/it][A
     58%|█████▊    | 729/1261 [19:05<13:47,  1.55s/it][A
     58%|█████▊    | 730/1261 [19:07<13:48,  1.56s/it][A
     58%|█████▊    | 731/1261 [19:08<13:47,  1.56s/it][A
     58%|█████▊    | 732/1261 [19:10<13:46,  1.56s/it][A
     58%|█████▊    | 733/1261 [19:11<13:41,  1.56s/it][A
     58%|█████▊    | 734/1261 [19:13<13:39,  1.55s/it][A
     58%|█████▊    | 735/1261 [19:14<13:35,  1.55s/it][A
     58%|█████▊    | 736/1261 [19:16<13:43,  1.57s/it][A
     58%|█████▊    | 737/1261 [19:17<13:43,  1.57s/it][A
     59%|█████▊    | 738/1261 [19:19<13:44,  1.58s/it][A
     59%|█████▊    | 739/1261 [19:21<13:37,  1.57s/it][A
     59%|█████▊    | 740/1261 [19:22<13:45,  1.58s/it][A
     59%|█████▉    | 741/1261 [19:24<13:42,  1.58s/it][A
     59%|█████▉    | 742/1261 [19:25<13:34,  1.57s/it][A
     59%|█████▉    | 743/1261 [19:27<13:28,  1.56s/it][A
     59%|█████▉    | 744/1261 [19:28<13:25,  1.56s/it][A
     59%|█████▉    | 745/1261 [19:30<13:20,  1.55s/it][A
     59%|█████▉    | 746/1261 [19:32<13:25,  1.56s/it][A
     59%|█████▉    | 747/1261 [19:33<13:23,  1.56s/it][A
     59%|█████▉    | 748/1261 [19:35<13:29,  1.58s/it][A
     59%|█████▉    | 749/1261 [19:36<13:22,  1.57s/it][A
     59%|█████▉    | 750/1261 [19:38<13:25,  1.58s/it][A
     60%|█████▉    | 751/1261 [19:39<13:18,  1.57s/it][A
     60%|█████▉    | 752/1261 [19:41<13:14,  1.56s/it][A
     60%|█████▉    | 753/1261 [19:43<13:17,  1.57s/it][A
     60%|█████▉    | 754/1261 [19:44<13:17,  1.57s/it][A
     60%|█████▉    | 755/1261 [19:46<13:16,  1.57s/it][A
     60%|█████▉    | 756/1261 [19:47<13:26,  1.60s/it][A
     60%|██████    | 757/1261 [19:49<13:20,  1.59s/it][A
     60%|██████    | 758/1261 [19:51<13:20,  1.59s/it][A
     60%|██████    | 759/1261 [19:52<13:13,  1.58s/it][A
     60%|██████    | 760/1261 [19:54<13:10,  1.58s/it][A
     60%|██████    | 761/1261 [19:55<13:23,  1.61s/it][A
     60%|██████    | 762/1261 [19:57<13:20,  1.60s/it][A
     61%|██████    | 763/1261 [19:58<13:11,  1.59s/it][A
     61%|██████    | 764/1261 [20:00<13:05,  1.58s/it][A
     61%|██████    | 765/1261 [20:02<12:56,  1.56s/it][A
     61%|██████    | 766/1261 [20:03<12:56,  1.57s/it][A
     61%|██████    | 767/1261 [20:05<12:56,  1.57s/it][A
     61%|██████    | 768/1261 [20:06<12:47,  1.56s/it][A
     61%|██████    | 769/1261 [20:08<12:44,  1.55s/it][A
     61%|██████    | 770/1261 [20:09<12:37,  1.54s/it][A
     61%|██████    | 771/1261 [20:11<12:36,  1.54s/it][A
     61%|██████    | 772/1261 [20:12<12:41,  1.56s/it][A
     61%|██████▏   | 773/1261 [20:14<12:41,  1.56s/it][A
     61%|██████▏   | 774/1261 [20:16<12:44,  1.57s/it][A
     61%|██████▏   | 775/1261 [20:17<12:43,  1.57s/it][A
     62%|██████▏   | 776/1261 [20:19<12:39,  1.57s/it][A
     62%|██████▏   | 777/1261 [20:20<12:33,  1.56s/it][A
     62%|██████▏   | 778/1261 [20:22<12:28,  1.55s/it][A
     62%|██████▏   | 779/1261 [20:23<12:31,  1.56s/it][A
     62%|██████▏   | 780/1261 [20:25<12:29,  1.56s/it][A
     62%|██████▏   | 781/1261 [20:26<12:27,  1.56s/it][A
     62%|██████▏   | 782/1261 [20:28<12:23,  1.55s/it][A
     62%|██████▏   | 783/1261 [20:30<12:21,  1.55s/it][A
     62%|██████▏   | 784/1261 [20:31<12:20,  1.55s/it][A
     62%|██████▏   | 785/1261 [20:33<12:27,  1.57s/it][A
     62%|██████▏   | 786/1261 [20:34<12:25,  1.57s/it][A
     62%|██████▏   | 787/1261 [20:36<12:26,  1.57s/it][A
     62%|██████▏   | 788/1261 [20:37<12:20,  1.57s/it][A
     63%|██████▎   | 789/1261 [20:39<12:22,  1.57s/it][A
     63%|██████▎   | 790/1261 [20:41<12:17,  1.57s/it][A
     63%|██████▎   | 791/1261 [20:42<12:17,  1.57s/it][A
     63%|██████▎   | 792/1261 [20:44<12:09,  1.56s/it][A
     63%|██████▎   | 793/1261 [20:45<12:06,  1.55s/it][A
     63%|██████▎   | 794/1261 [20:47<12:12,  1.57s/it][A
     63%|██████▎   | 795/1261 [20:48<12:14,  1.58s/it][A
     63%|██████▎   | 796/1261 [20:50<12:21,  1.59s/it][A
     63%|██████▎   | 797/1261 [20:52<12:18,  1.59s/it][A
     63%|██████▎   | 798/1261 [20:53<12:14,  1.59s/it][A
     63%|██████▎   | 799/1261 [20:55<12:05,  1.57s/it][A
     63%|██████▎   | 800/1261 [20:56<12:05,  1.57s/it][A
     64%|██████▎   | 801/1261 [20:58<12:05,  1.58s/it][A
     64%|██████▎   | 802/1261 [21:00<12:07,  1.58s/it][A
     64%|██████▎   | 803/1261 [21:01<12:01,  1.57s/it][A
     64%|██████▍   | 804/1261 [21:03<11:55,  1.57s/it][A
     64%|██████▍   | 805/1261 [21:04<11:50,  1.56s/it][A
     64%|██████▍   | 806/1261 [21:06<11:50,  1.56s/it][A
     64%|██████▍   | 807/1261 [21:07<11:50,  1.57s/it][A
     64%|██████▍   | 808/1261 [21:09<12:01,  1.59s/it][A
     64%|██████▍   | 809/1261 [21:11<11:55,  1.58s/it][A
     64%|██████▍   | 810/1261 [21:12<11:49,  1.57s/it][A
     64%|██████▍   | 811/1261 [21:14<11:45,  1.57s/it][A
     64%|██████▍   | 812/1261 [21:15<11:47,  1.58s/it][A
     64%|██████▍   | 813/1261 [21:17<11:44,  1.57s/it][A
     65%|██████▍   | 814/1261 [21:18<11:45,  1.58s/it][A
     65%|██████▍   | 815/1261 [21:20<11:40,  1.57s/it][A
     65%|██████▍   | 816/1261 [21:22<11:37,  1.57s/it][A
     65%|██████▍   | 817/1261 [21:23<11:34,  1.56s/it][A
     65%|██████▍   | 818/1261 [21:25<11:38,  1.58s/it][A
     65%|██████▍   | 819/1261 [21:26<11:39,  1.58s/it][A
     65%|██████▌   | 820/1261 [21:28<11:45,  1.60s/it][A
     65%|██████▌   | 821/1261 [21:29<11:43,  1.60s/it][A
     65%|██████▌   | 822/1261 [21:31<11:38,  1.59s/it][A
     65%|██████▌   | 823/1261 [21:33<11:35,  1.59s/it][A
     65%|██████▌   | 824/1261 [21:34<11:33,  1.59s/it][A
     65%|██████▌   | 825/1261 [21:36<11:28,  1.58s/it][A
     66%|██████▌   | 826/1261 [21:37<11:28,  1.58s/it][A
     66%|██████▌   | 827/1261 [21:39<11:30,  1.59s/it][A
     66%|██████▌   | 828/1261 [21:41<11:26,  1.59s/it][A
     66%|██████▌   | 829/1261 [21:42<11:28,  1.59s/it][A
     66%|██████▌   | 830/1261 [21:44<11:27,  1.60s/it][A
     66%|██████▌   | 831/1261 [21:45<11:27,  1.60s/it][A
     66%|██████▌   | 832/1261 [21:47<11:38,  1.63s/it][A
     66%|██████▌   | 833/1261 [21:49<11:39,  1.63s/it][A
     66%|██████▌   | 834/1261 [21:50<11:34,  1.63s/it][A
     66%|██████▌   | 835/1261 [21:52<11:24,  1.61s/it][A
     66%|██████▋   | 836/1261 [21:54<11:28,  1.62s/it][A
     66%|██████▋   | 837/1261 [21:55<11:23,  1.61s/it][A
     66%|██████▋   | 838/1261 [21:57<11:18,  1.60s/it][A
     67%|██████▋   | 839/1261 [21:58<11:12,  1.59s/it][A
     67%|██████▋   | 840/1261 [22:00<11:04,  1.58s/it][A
     67%|██████▋   | 841/1261 [22:01<11:02,  1.58s/it][A
     67%|██████▋   | 842/1261 [22:03<11:02,  1.58s/it][A
     67%|██████▋   | 843/1261 [22:05<11:00,  1.58s/it][A
     67%|██████▋   | 844/1261 [22:06<10:56,  1.57s/it][A
     67%|██████▋   | 845/1261 [22:08<10:56,  1.58s/it][A
     67%|██████▋   | 846/1261 [22:09<10:54,  1.58s/it][A
     67%|██████▋   | 847/1261 [22:11<10:52,  1.58s/it][A
     67%|██████▋   | 848/1261 [22:12<10:48,  1.57s/it][A
     67%|██████▋   | 849/1261 [22:14<10:47,  1.57s/it][A
     67%|██████▋   | 850/1261 [22:16<10:44,  1.57s/it][A
     67%|██████▋   | 851/1261 [22:17<10:44,  1.57s/it][A
     68%|██████▊   | 852/1261 [22:19<10:40,  1.57s/it][A
     68%|██████▊   | 853/1261 [22:20<10:48,  1.59s/it][A
     68%|██████▊   | 854/1261 [22:22<10:45,  1.59s/it][A
     68%|██████▊   | 855/1261 [22:24<10:51,  1.60s/it][A
     68%|██████▊   | 856/1261 [22:25<10:50,  1.60s/it][A
     68%|██████▊   | 857/1261 [22:27<10:45,  1.60s/it][A
     68%|██████▊   | 858/1261 [22:28<10:46,  1.60s/it][A
     68%|██████▊   | 859/1261 [22:30<10:37,  1.58s/it][A
     68%|██████▊   | 860/1261 [22:32<10:39,  1.59s/it][A
     68%|██████▊   | 861/1261 [22:33<10:33,  1.58s/it][A
     68%|██████▊   | 862/1261 [22:35<10:31,  1.58s/it][A
     68%|██████▊   | 863/1261 [22:36<10:23,  1.57s/it][A
     69%|██████▊   | 864/1261 [22:38<10:18,  1.56s/it][A
     69%|██████▊   | 865/1261 [22:39<10:17,  1.56s/it][A
     69%|██████▊   | 866/1261 [22:41<10:22,  1.58s/it][A
     69%|██████▉   | 867/1261 [22:43<10:24,  1.59s/it][A
     69%|██████▉   | 868/1261 [22:44<10:23,  1.59s/it][A
     69%|██████▉   | 869/1261 [22:46<10:20,  1.58s/it][A
     69%|██████▉   | 870/1261 [22:47<10:19,  1.58s/it][A
     69%|██████▉   | 871/1261 [22:49<10:16,  1.58s/it][A
     69%|██████▉   | 872/1261 [22:50<10:11,  1.57s/it][A
     69%|██████▉   | 873/1261 [22:52<10:05,  1.56s/it][A
     69%|██████▉   | 874/1261 [22:53<10:02,  1.56s/it][A
     69%|██████▉   | 875/1261 [22:55<09:59,  1.55s/it][A
     69%|██████▉   | 876/1261 [22:57<10:01,  1.56s/it][A
     70%|██████▉   | 877/1261 [22:58<10:00,  1.56s/it][A
     70%|██████▉   | 878/1261 [23:00<10:02,  1.57s/it][A
     70%|██████▉   | 879/1261 [23:01<09:58,  1.57s/it][A
     70%|██████▉   | 880/1261 [23:03<10:05,  1.59s/it][A
     70%|██████▉   | 881/1261 [23:05<10:03,  1.59s/it][A
     70%|██████▉   | 882/1261 [23:06<10:02,  1.59s/it][A
     70%|███████   | 883/1261 [23:08<09:59,  1.59s/it][A
     70%|███████   | 884/1261 [23:09<10:02,  1.60s/it][A
     70%|███████   | 885/1261 [23:11<10:00,  1.60s/it][A
     70%|███████   | 886/1261 [23:13<10:00,  1.60s/it][A
     70%|███████   | 887/1261 [23:14<09:56,  1.59s/it][A
     70%|███████   | 888/1261 [23:16<09:51,  1.59s/it][A
     70%|███████   | 889/1261 [23:17<09:49,  1.59s/it][A
     71%|███████   | 890/1261 [23:19<09:47,  1.58s/it][A
     71%|███████   | 891/1261 [23:20<09:45,  1.58s/it][A
     71%|███████   | 892/1261 [23:22<09:47,  1.59s/it][A
     71%|███████   | 893/1261 [23:24<09:44,  1.59s/it][A
     71%|███████   | 894/1261 [23:25<09:39,  1.58s/it][A
     71%|███████   | 895/1261 [23:27<09:34,  1.57s/it][A
     71%|███████   | 896/1261 [23:28<09:30,  1.56s/it][A
     71%|███████   | 897/1261 [23:30<09:30,  1.57s/it][A
     71%|███████   | 898/1261 [23:31<09:34,  1.58s/it][A
     71%|███████▏  | 899/1261 [23:33<09:28,  1.57s/it][A
     71%|███████▏  | 900/1261 [23:35<09:23,  1.56s/it][A
     71%|███████▏  | 901/1261 [23:36<09:26,  1.57s/it][A
     72%|███████▏  | 902/1261 [23:38<09:27,  1.58s/it][A
     72%|███████▏  | 903/1261 [23:39<09:28,  1.59s/it][A
     72%|███████▏  | 904/1261 [23:41<09:24,  1.58s/it][A
     72%|███████▏  | 905/1261 [23:43<09:25,  1.59s/it][A
     72%|███████▏  | 906/1261 [23:44<09:22,  1.58s/it][A
     72%|███████▏  | 907/1261 [23:46<09:25,  1.60s/it][A
     72%|███████▏  | 908/1261 [23:47<09:25,  1.60s/it][A
     72%|███████▏  | 909/1261 [23:49<09:29,  1.62s/it][A
     72%|███████▏  | 910/1261 [23:51<09:21,  1.60s/it][A
     72%|███████▏  | 911/1261 [23:52<09:24,  1.61s/it][A
     72%|███████▏  | 912/1261 [23:54<09:24,  1.62s/it][A
     72%|███████▏  | 913/1261 [23:55<09:23,  1.62s/it][A
     72%|███████▏  | 914/1261 [23:57<09:22,  1.62s/it][A
     73%|███████▎  | 915/1261 [23:59<09:20,  1.62s/it][A
     73%|███████▎  | 916/1261 [24:00<09:19,  1.62s/it][A
     73%|███████▎  | 917/1261 [24:02<09:14,  1.61s/it][A
     73%|███████▎  | 918/1261 [24:03<09:06,  1.59s/it][A
     73%|███████▎  | 919/1261 [24:05<09:03,  1.59s/it][A
     73%|███████▎  | 920/1261 [24:07<09:01,  1.59s/it][A
     73%|███████▎  | 921/1261 [24:08<08:55,  1.57s/it][A
     73%|███████▎  | 922/1261 [24:10<08:57,  1.58s/it][A
     73%|███████▎  | 923/1261 [24:11<08:52,  1.58s/it][A
     73%|███████▎  | 924/1261 [24:13<08:47,  1.57s/it][A
     73%|███████▎  | 925/1261 [24:14<08:44,  1.56s/it][A
     73%|███████▎  | 926/1261 [24:16<08:46,  1.57s/it][A
     74%|███████▎  | 927/1261 [24:18<08:46,  1.58s/it][A
     74%|███████▎  | 928/1261 [24:19<08:45,  1.58s/it][A
     74%|███████▎  | 929/1261 [24:21<08:39,  1.57s/it][A
     74%|███████▍  | 930/1261 [24:22<08:41,  1.58s/it][A
     74%|███████▍  | 931/1261 [24:24<08:41,  1.58s/it][A
     74%|███████▍  | 932/1261 [24:26<08:39,  1.58s/it][A
     74%|███████▍  | 933/1261 [24:27<08:35,  1.57s/it][A
     74%|███████▍  | 934/1261 [24:29<08:30,  1.56s/it][A
     74%|███████▍  | 935/1261 [24:30<08:29,  1.56s/it][A
     74%|███████▍  | 936/1261 [24:32<08:27,  1.56s/it][A
     74%|███████▍  | 937/1261 [24:33<08:26,  1.56s/it][A
     74%|███████▍  | 938/1261 [24:35<08:28,  1.57s/it][A
     74%|███████▍  | 939/1261 [24:36<08:29,  1.58s/it][A
     75%|███████▍  | 940/1261 [24:38<08:26,  1.58s/it][A
     75%|███████▍  | 941/1261 [24:40<08:21,  1.57s/it][A
     75%|███████▍  | 942/1261 [24:41<08:20,  1.57s/it][A
     75%|███████▍  | 943/1261 [24:43<08:19,  1.57s/it][A
     75%|███████▍  | 944/1261 [24:44<08:16,  1.57s/it][A
     75%|███████▍  | 945/1261 [24:46<08:14,  1.56s/it][A
     75%|███████▌  | 946/1261 [24:47<08:09,  1.55s/it][A
     75%|███████▌  | 947/1261 [24:49<08:09,  1.56s/it][A
     75%|███████▌  | 948/1261 [24:51<08:09,  1.57s/it][A
     75%|███████▌  | 949/1261 [24:52<08:17,  1.59s/it][A
     75%|███████▌  | 950/1261 [24:54<08:21,  1.61s/it][A
     75%|███████▌  | 951/1261 [24:55<08:17,  1.61s/it][A
     75%|███████▌  | 952/1261 [24:57<08:10,  1.59s/it][A
     76%|███████▌  | 953/1261 [24:59<08:07,  1.58s/it][A
     76%|███████▌  | 954/1261 [25:00<08:05,  1.58s/it][A
     76%|███████▌  | 955/1261 [25:02<08:06,  1.59s/it][A
     76%|███████▌  | 956/1261 [25:03<08:04,  1.59s/it][A
     76%|███████▌  | 957/1261 [25:05<08:04,  1.59s/it][A
     76%|███████▌  | 958/1261 [25:07<08:03,  1.60s/it][A
     76%|███████▌  | 959/1261 [25:08<07:57,  1.58s/it][A
     76%|███████▌  | 960/1261 [25:10<07:56,  1.58s/it][A
     76%|███████▌  | 961/1261 [25:11<07:55,  1.58s/it][A
     76%|███████▋  | 962/1261 [25:12<07:03,  1.42s/it][A
     76%|███████▋  | 963/1261 [25:13<06:23,  1.29s/it][A
     76%|███████▋  | 964/1261 [25:14<05:57,  1.20s/it][A
     77%|███████▋  | 965/1261 [25:16<06:24,  1.30s/it][A
     77%|███████▋  | 966/1261 [25:17<06:47,  1.38s/it][A
     77%|███████▋  | 967/1261 [25:19<07:00,  1.43s/it][A
     77%|███████▋  | 968/1261 [25:21<07:11,  1.47s/it][A
     77%|███████▋  | 969/1261 [25:22<07:14,  1.49s/it][A
     77%|███████▋  | 970/1261 [25:24<07:18,  1.51s/it][A
     77%|███████▋  | 971/1261 [25:25<07:19,  1.51s/it][A
     77%|███████▋  | 972/1261 [25:27<07:21,  1.53s/it][A
     77%|███████▋  | 973/1261 [25:28<07:20,  1.53s/it][A
     77%|███████▋  | 974/1261 [25:30<07:16,  1.52s/it][A
     77%|███████▋  | 975/1261 [25:31<07:13,  1.51s/it][A
     77%|███████▋  | 976/1261 [25:33<07:09,  1.51s/it][A
     77%|███████▋  | 977/1261 [25:34<07:10,  1.52s/it][A
     78%|███████▊  | 978/1261 [25:36<07:04,  1.50s/it][A
     78%|███████▊  | 979/1261 [25:37<06:57,  1.48s/it][A
     78%|███████▊  | 980/1261 [25:39<06:55,  1.48s/it][A
     78%|███████▊  | 981/1261 [25:40<06:51,  1.47s/it][A
     78%|███████▊  | 982/1261 [25:41<06:46,  1.46s/it][A
     78%|███████▊  | 983/1261 [25:43<06:48,  1.47s/it][A
     78%|███████▊  | 984/1261 [25:44<06:49,  1.48s/it][A
     78%|███████▊  | 985/1261 [25:46<07:09,  1.55s/it][A
     78%|███████▊  | 986/1261 [25:48<07:12,  1.57s/it][A
     78%|███████▊  | 987/1261 [25:49<07:13,  1.58s/it][A
     78%|███████▊  | 988/1261 [25:51<07:08,  1.57s/it][A
     78%|███████▊  | 989/1261 [25:53<07:22,  1.63s/it][A
     79%|███████▊  | 990/1261 [25:54<07:20,  1.63s/it][A
     79%|███████▊  | 991/1261 [25:56<07:23,  1.64s/it][A
     79%|███████▊  | 992/1261 [25:58<07:25,  1.66s/it][A
     79%|███████▊  | 993/1261 [25:59<07:28,  1.68s/it][A
     79%|███████▉  | 994/1261 [26:01<07:32,  1.69s/it][A
     79%|███████▉  | 995/1261 [26:03<07:35,  1.71s/it][A
     79%|███████▉  | 996/1261 [26:05<07:42,  1.74s/it][A
     79%|███████▉  | 997/1261 [26:07<07:44,  1.76s/it][A
     79%|███████▉  | 998/1261 [26:08<07:50,  1.79s/it][A
     79%|███████▉  | 999/1261 [26:10<07:54,  1.81s/it][A
     79%|███████▉  | 1000/1261 [26:12<07:58,  1.83s/it][A
     79%|███████▉  | 1001/1261 [26:14<07:56,  1.83s/it][A
     79%|███████▉  | 1002/1261 [26:16<07:50,  1.82s/it][A
     80%|███████▉  | 1003/1261 [26:18<07:49,  1.82s/it][A
     80%|███████▉  | 1004/1261 [26:19<07:45,  1.81s/it][A
     80%|███████▉  | 1005/1261 [26:21<07:41,  1.80s/it][A
     80%|███████▉  | 1006/1261 [26:23<07:32,  1.78s/it][A
     80%|███████▉  | 1007/1261 [26:25<07:25,  1.76s/it][A
     80%|███████▉  | 1008/1261 [26:26<07:18,  1.73s/it][A
     80%|████████  | 1009/1261 [26:28<07:10,  1.71s/it][A
     80%|████████  | 1010/1261 [26:30<07:09,  1.71s/it][A
     80%|████████  | 1011/1261 [26:31<07:04,  1.70s/it][A
     80%|████████  | 1012/1261 [26:33<07:02,  1.70s/it][A
     80%|████████  | 1013/1261 [26:35<07:01,  1.70s/it][A
     80%|████████  | 1014/1261 [26:36<06:57,  1.69s/it][A
     80%|████████  | 1015/1261 [26:38<06:54,  1.69s/it][A
     81%|████████  | 1016/1261 [26:40<06:49,  1.67s/it][A
     81%|████████  | 1017/1261 [26:41<06:49,  1.68s/it][A
     81%|████████  | 1018/1261 [26:43<06:47,  1.68s/it][A
     81%|████████  | 1019/1261 [26:45<06:43,  1.67s/it][A
     81%|████████  | 1020/1261 [26:46<06:41,  1.67s/it][A
     81%|████████  | 1021/1261 [26:48<06:37,  1.66s/it][A
     81%|████████  | 1022/1261 [26:50<06:36,  1.66s/it][A
     81%|████████  | 1023/1261 [26:51<06:33,  1.65s/it][A
     81%|████████  | 1024/1261 [26:53<06:32,  1.66s/it][A
     81%|████████▏ | 1025/1261 [26:55<06:30,  1.65s/it][A
     81%|████████▏ | 1026/1261 [26:56<06:27,  1.65s/it][A
     81%|████████▏ | 1027/1261 [26:58<06:20,  1.63s/it][A
     82%|████████▏ | 1028/1261 [26:59<06:17,  1.62s/it][A
     82%|████████▏ | 1029/1261 [27:01<06:12,  1.60s/it][A
     82%|████████▏ | 1030/1261 [27:03<06:07,  1.59s/it][A
     82%|████████▏ | 1031/1261 [27:04<06:01,  1.57s/it][A
     82%|████████▏ | 1032/1261 [27:06<05:56,  1.56s/it][A
     82%|████████▏ | 1033/1261 [27:07<05:51,  1.54s/it][A
     82%|████████▏ | 1034/1261 [27:09<05:45,  1.52s/it][A
     82%|████████▏ | 1035/1261 [27:10<05:39,  1.50s/it][A
     82%|████████▏ | 1036/1261 [27:11<05:32,  1.48s/it][A
     82%|████████▏ | 1037/1261 [27:13<05:26,  1.46s/it][A
     82%|████████▏ | 1038/1261 [27:14<05:22,  1.45s/it][A
     82%|████████▏ | 1039/1261 [27:16<05:16,  1.43s/it][A
     82%|████████▏ | 1040/1261 [27:17<05:12,  1.42s/it][A
     83%|████████▎ | 1041/1261 [27:18<05:09,  1.41s/it][A
     83%|████████▎ | 1042/1261 [27:20<05:07,  1.40s/it][A
     83%|████████▎ | 1043/1261 [27:21<05:02,  1.39s/it][A
     83%|████████▎ | 1044/1261 [27:23<05:00,  1.39s/it][A
     83%|████████▎ | 1045/1261 [27:24<05:00,  1.39s/it][A
     83%|████████▎ | 1046/1261 [27:25<05:02,  1.41s/it][A
     83%|████████▎ | 1047/1261 [27:27<05:00,  1.41s/it][A
     83%|████████▎ | 1048/1261 [27:28<05:01,  1.42s/it][A
     83%|████████▎ | 1049/1261 [27:30<05:00,  1.42s/it][A
     83%|████████▎ | 1050/1261 [27:31<05:04,  1.44s/it][A
     83%|████████▎ | 1051/1261 [27:33<05:08,  1.47s/it][A
     83%|████████▎ | 1052/1261 [27:34<05:11,  1.49s/it][A
     84%|████████▎ | 1053/1261 [27:36<05:13,  1.51s/it][A
     84%|████████▎ | 1054/1261 [27:37<05:13,  1.51s/it][A
     84%|████████▎ | 1055/1261 [27:39<05:13,  1.52s/it][A
     84%|████████▎ | 1056/1261 [27:40<05:14,  1.53s/it][A
     84%|████████▍ | 1057/1261 [27:42<05:16,  1.55s/it][A
     84%|████████▍ | 1058/1261 [27:44<05:16,  1.56s/it][A
     84%|████████▍ | 1059/1261 [27:45<05:14,  1.56s/it][A
     84%|████████▍ | 1060/1261 [27:47<05:17,  1.58s/it][A
     84%|████████▍ | 1061/1261 [27:48<05:16,  1.58s/it][A
     84%|████████▍ | 1062/1261 [27:50<05:13,  1.58s/it][A
     84%|████████▍ | 1063/1261 [27:52<05:19,  1.62s/it][A
     84%|████████▍ | 1064/1261 [27:53<05:18,  1.62s/it][A
     84%|████████▍ | 1065/1261 [27:55<05:14,  1.60s/it][A
     85%|████████▍ | 1066/1261 [27:56<05:09,  1.59s/it][A
     85%|████████▍ | 1067/1261 [27:58<05:07,  1.58s/it][A
     85%|████████▍ | 1068/1261 [28:00<05:04,  1.58s/it][A
     85%|████████▍ | 1069/1261 [28:01<05:02,  1.57s/it][A
     85%|████████▍ | 1070/1261 [28:03<05:00,  1.57s/it][A
     85%|████████▍ | 1071/1261 [28:04<05:01,  1.58s/it][A
     85%|████████▌ | 1072/1261 [28:06<04:57,  1.57s/it][A
     85%|████████▌ | 1073/1261 [28:07<04:55,  1.57s/it][A
     85%|████████▌ | 1074/1261 [28:09<04:55,  1.58s/it][A
     85%|████████▌ | 1075/1261 [28:11<04:55,  1.59s/it][A
     85%|████████▌ | 1076/1261 [28:12<04:54,  1.59s/it][A
     85%|████████▌ | 1077/1261 [28:14<04:51,  1.58s/it][A
     85%|████████▌ | 1078/1261 [28:15<04:47,  1.57s/it][A
     86%|████████▌ | 1079/1261 [28:17<04:48,  1.58s/it][A
     86%|████████▌ | 1080/1261 [28:19<04:47,  1.59s/it][A
     86%|████████▌ | 1081/1261 [28:20<04:45,  1.59s/it][A
     86%|████████▌ | 1082/1261 [28:22<04:43,  1.58s/it][A
     86%|████████▌ | 1083/1261 [28:23<04:40,  1.58s/it][A
     86%|████████▌ | 1084/1261 [28:25<04:37,  1.57s/it][A
     86%|████████▌ | 1085/1261 [28:26<04:37,  1.58s/it][A
     86%|████████▌ | 1086/1261 [28:28<04:36,  1.58s/it][A
     86%|████████▌ | 1087/1261 [28:30<04:33,  1.57s/it][A
     86%|████████▋ | 1088/1261 [28:31<04:29,  1.56s/it][A
     86%|████████▋ | 1089/1261 [28:33<04:32,  1.58s/it][A
     86%|████████▋ | 1090/1261 [28:34<04:33,  1.60s/it][A
     87%|████████▋ | 1091/1261 [28:36<04:31,  1.60s/it][A
     87%|████████▋ | 1092/1261 [28:38<04:29,  1.60s/it][A
     87%|████████▋ | 1093/1261 [28:39<04:25,  1.58s/it][A
     87%|████████▋ | 1094/1261 [28:41<04:21,  1.56s/it][A
     87%|████████▋ | 1095/1261 [28:42<04:22,  1.58s/it][A
     87%|████████▋ | 1096/1261 [28:44<04:19,  1.57s/it][A
     87%|████████▋ | 1097/1261 [28:45<04:16,  1.57s/it][A
     87%|████████▋ | 1098/1261 [28:47<04:14,  1.56s/it][A
     87%|████████▋ | 1099/1261 [28:48<04:13,  1.57s/it][A
     87%|████████▋ | 1100/1261 [28:50<04:10,  1.56s/it][A
     87%|████████▋ | 1101/1261 [28:52<04:10,  1.57s/it][A
     87%|████████▋ | 1102/1261 [28:53<04:07,  1.56s/it][A
     87%|████████▋ | 1103/1261 [28:55<04:06,  1.56s/it][A
     88%|████████▊ | 1104/1261 [28:56<04:08,  1.58s/it][A
     88%|████████▊ | 1105/1261 [28:58<04:07,  1.58s/it][A
     88%|████████▊ | 1106/1261 [28:59<04:04,  1.58s/it][A
     88%|████████▊ | 1107/1261 [29:01<04:03,  1.58s/it][A
     88%|████████▊ | 1108/1261 [29:03<04:00,  1.57s/it][A
     88%|████████▊ | 1109/1261 [29:04<04:00,  1.58s/it][A
     88%|████████▊ | 1110/1261 [29:06<03:59,  1.58s/it][A
     88%|████████▊ | 1111/1261 [29:07<03:57,  1.58s/it][A
     88%|████████▊ | 1112/1261 [29:09<03:53,  1.57s/it][A
     88%|████████▊ | 1113/1261 [29:10<03:51,  1.56s/it][A
     88%|████████▊ | 1114/1261 [29:12<03:50,  1.56s/it][A
     88%|████████▊ | 1115/1261 [29:14<03:51,  1.59s/it][A
     89%|████████▊ | 1116/1261 [29:15<03:49,  1.59s/it][A
     89%|████████▊ | 1117/1261 [29:17<03:49,  1.60s/it][A
     89%|████████▊ | 1118/1261 [29:18<03:46,  1.58s/it][A
     89%|████████▊ | 1119/1261 [29:20<03:44,  1.58s/it][A
     89%|████████▉ | 1120/1261 [29:22<03:43,  1.58s/it][A
     89%|████████▉ | 1121/1261 [29:23<03:41,  1.58s/it][A
     89%|████████▉ | 1122/1261 [29:25<03:33,  1.54s/it][A
     89%|████████▉ | 1123/1261 [29:26<03:10,  1.38s/it][A
     89%|████████▉ | 1124/1261 [29:27<02:52,  1.26s/it][A
     89%|████████▉ | 1125/1261 [29:28<02:42,  1.20s/it][A
     89%|████████▉ | 1126/1261 [29:29<02:56,  1.31s/it][A
     89%|████████▉ | 1127/1261 [29:31<03:08,  1.41s/it][A
     89%|████████▉ | 1128/1261 [29:32<03:15,  1.47s/it][A
     90%|████████▉ | 1129/1261 [29:34<03:18,  1.50s/it][A
     90%|████████▉ | 1130/1261 [29:36<03:18,  1.52s/it][A
     90%|████████▉ | 1131/1261 [29:37<03:18,  1.53s/it][A
     90%|████████▉ | 1132/1261 [29:39<03:18,  1.54s/it][A
     90%|████████▉ | 1133/1261 [29:40<03:19,  1.56s/it][A
     90%|████████▉ | 1134/1261 [29:42<03:19,  1.57s/it][A
     90%|█████████ | 1135/1261 [29:43<03:17,  1.56s/it][A
     90%|█████████ | 1136/1261 [29:45<03:16,  1.57s/it][A
     90%|█████████ | 1137/1261 [29:47<03:20,  1.62s/it][A
     90%|█████████ | 1138/1261 [29:48<03:23,  1.65s/it][A
     90%|█████████ | 1139/1261 [29:50<03:19,  1.64s/it][A
     90%|█████████ | 1140/1261 [29:52<03:18,  1.64s/it][A
     90%|█████████ | 1141/1261 [29:53<03:13,  1.61s/it][A
     91%|█████████ | 1142/1261 [29:55<03:09,  1.59s/it][A
     91%|█████████ | 1143/1261 [29:56<03:06,  1.58s/it][A
     91%|█████████ | 1144/1261 [29:58<03:03,  1.57s/it][A
     91%|█████████ | 1145/1261 [29:59<03:00,  1.56s/it][A
     91%|█████████ | 1146/1261 [30:01<03:00,  1.57s/it][A
     91%|█████████ | 1147/1261 [30:03<03:00,  1.59s/it][A
     91%|█████████ | 1148/1261 [30:04<03:01,  1.60s/it][A
     91%|█████████ | 1149/1261 [30:06<02:58,  1.60s/it][A
     91%|█████████ | 1150/1261 [30:08<02:57,  1.60s/it][A
     91%|█████████▏| 1151/1261 [30:09<02:55,  1.59s/it][A
     91%|█████████▏| 1152/1261 [30:11<02:52,  1.58s/it][A
     91%|█████████▏| 1153/1261 [30:12<02:51,  1.59s/it][A
     92%|█████████▏| 1154/1261 [30:14<02:48,  1.58s/it][A
     92%|█████████▏| 1155/1261 [30:15<02:46,  1.57s/it][A
     92%|█████████▏| 1156/1261 [30:17<02:44,  1.57s/it][A
     92%|█████████▏| 1157/1261 [30:18<02:43,  1.57s/it][A
     92%|█████████▏| 1158/1261 [30:20<02:42,  1.58s/it][A
     92%|█████████▏| 1159/1261 [30:22<02:40,  1.57s/it][A
     92%|█████████▏| 1160/1261 [30:23<02:39,  1.58s/it][A
     92%|█████████▏| 1161/1261 [30:25<02:36,  1.57s/it][A
     92%|█████████▏| 1162/1261 [30:26<02:35,  1.57s/it][A
     92%|█████████▏| 1163/1261 [30:28<02:38,  1.61s/it][A
     92%|█████████▏| 1164/1261 [30:30<02:39,  1.64s/it][A
     92%|█████████▏| 1165/1261 [30:31<02:35,  1.62s/it][A
     92%|█████████▏| 1166/1261 [30:33<02:34,  1.63s/it][A
     93%|█████████▎| 1167/1261 [30:35<02:31,  1.61s/it][A
     93%|█████████▎| 1168/1261 [30:36<02:29,  1.61s/it][A
     93%|█████████▎| 1169/1261 [30:38<02:33,  1.67s/it][A
     93%|█████████▎| 1170/1261 [30:40<02:30,  1.66s/it][A
     93%|█████████▎| 1171/1261 [30:41<02:26,  1.62s/it][A
     93%|█████████▎| 1172/1261 [30:43<02:26,  1.65s/it][A
     93%|█████████▎| 1173/1261 [30:44<02:23,  1.63s/it][A
     93%|█████████▎| 1174/1261 [30:46<02:19,  1.60s/it][A
     93%|█████████▎| 1175/1261 [30:48<02:17,  1.60s/it][A
     93%|█████████▎| 1176/1261 [30:49<02:19,  1.64s/it][A
     93%|█████████▎| 1177/1261 [30:51<02:20,  1.67s/it][A
     93%|█████████▎| 1178/1261 [30:53<02:31,  1.83s/it][A
     93%|█████████▎| 1179/1261 [30:55<02:30,  1.84s/it][A
     94%|█████████▎| 1180/1261 [30:57<02:22,  1.76s/it][A
     94%|█████████▎| 1181/1261 [30:58<02:15,  1.70s/it][A
     94%|█████████▎| 1182/1261 [31:00<02:12,  1.67s/it][A
     94%|█████████▍| 1183/1261 [31:01<02:08,  1.65s/it][A
     94%|█████████▍| 1184/1261 [31:03<02:03,  1.61s/it][A
     94%|█████████▍| 1185/1261 [31:05<02:02,  1.61s/it][A
     94%|█████████▍| 1186/1261 [31:07<02:11,  1.75s/it][A
     94%|█████████▍| 1187/1261 [31:08<02:05,  1.69s/it][A
     94%|█████████▍| 1188/1261 [31:10<02:09,  1.77s/it][A
     94%|█████████▍| 1189/1261 [31:12<02:06,  1.76s/it][A
     94%|█████████▍| 1190/1261 [31:14<02:05,  1.77s/it][A
     94%|█████████▍| 1191/1261 [31:16<02:09,  1.85s/it][A
     95%|█████████▍| 1192/1261 [31:18<02:12,  1.92s/it][A
     95%|█████████▍| 1193/1261 [31:20<02:13,  1.96s/it][A
     95%|█████████▍| 1194/1261 [31:22<02:08,  1.92s/it][A
     95%|█████████▍| 1195/1261 [31:24<02:04,  1.89s/it][A
     95%|█████████▍| 1196/1261 [31:25<02:04,  1.91s/it][A
     95%|█████████▍| 1197/1261 [31:27<01:59,  1.87s/it][A
     95%|█████████▌| 1198/1261 [31:29<01:57,  1.86s/it][A
     95%|█████████▌| 1199/1261 [31:31<01:52,  1.81s/it][A
     95%|█████████▌| 1200/1261 [31:33<01:51,  1.82s/it][A
     95%|█████████▌| 1201/1261 [31:34<01:45,  1.76s/it][A
     95%|█████████▌| 1202/1261 [31:36<01:46,  1.80s/it][A
     95%|█████████▌| 1203/1261 [31:38<01:48,  1.87s/it][A
     95%|█████████▌| 1204/1261 [31:40<01:46,  1.87s/it][A
     96%|█████████▌| 1205/1261 [31:42<01:43,  1.85s/it][A
     96%|█████████▌| 1206/1261 [31:44<01:41,  1.85s/it][A
     96%|█████████▌| 1207/1261 [31:46<01:41,  1.88s/it][A
     96%|█████████▌| 1208/1261 [31:48<01:43,  1.96s/it][A
     96%|█████████▌| 1209/1261 [31:50<01:44,  2.02s/it][A
     96%|█████████▌| 1210/1261 [31:52<01:43,  2.03s/it][A
     96%|█████████▌| 1211/1261 [31:54<01:44,  2.09s/it][A
     96%|█████████▌| 1212/1261 [31:56<01:40,  2.06s/it][A
     96%|█████████▌| 1213/1261 [31:58<01:37,  2.02s/it][A
     96%|█████████▋| 1214/1261 [32:00<01:33,  1.99s/it][A
     96%|█████████▋| 1215/1261 [32:02<01:31,  1.99s/it][A
     96%|█████████▋| 1216/1261 [32:04<01:31,  2.04s/it][A
     97%|█████████▋| 1217/1261 [32:06<01:27,  1.99s/it][A
     97%|█████████▋| 1218/1261 [32:08<01:24,  1.97s/it][A
     97%|█████████▋| 1219/1261 [32:10<01:22,  1.96s/it][A
     97%|█████████▋| 1220/1261 [32:12<01:20,  1.96s/it][A
     97%|█████████▋| 1221/1261 [32:14<01:17,  1.94s/it][A
     97%|█████████▋| 1222/1261 [32:16<01:13,  1.89s/it][A
     97%|█████████▋| 1223/1261 [32:17<01:11,  1.88s/it][A
     97%|█████████▋| 1224/1261 [32:19<01:08,  1.86s/it][A
     97%|█████████▋| 1225/1261 [32:21<01:07,  1.88s/it][A
     97%|█████████▋| 1226/1261 [32:23<01:04,  1.85s/it][A
     97%|█████████▋| 1227/1261 [32:25<01:01,  1.82s/it][A
     97%|█████████▋| 1228/1261 [32:27<01:00,  1.85s/it][A
     97%|█████████▋| 1229/1261 [32:29<01:00,  1.88s/it][A
     98%|█████████▊| 1230/1261 [32:30<00:57,  1.85s/it][A
     98%|█████████▊| 1231/1261 [32:32<00:54,  1.82s/it][A
     98%|█████████▊| 1232/1261 [32:34<00:51,  1.77s/it][A
     98%|█████████▊| 1233/1261 [32:35<00:48,  1.73s/it][A
     98%|█████████▊| 1234/1261 [32:37<00:46,  1.74s/it][A
     98%|█████████▊| 1235/1261 [32:39<00:45,  1.76s/it][A
     98%|█████████▊| 1236/1261 [32:41<00:44,  1.77s/it][A
     98%|█████████▊| 1237/1261 [32:42<00:42,  1.77s/it][A
     98%|█████████▊| 1238/1261 [32:44<00:40,  1.77s/it][A
     98%|█████████▊| 1239/1261 [32:46<00:39,  1.79s/it][A
     98%|█████████▊| 1240/1261 [32:48<00:37,  1.81s/it][A
     98%|█████████▊| 1241/1261 [32:50<00:35,  1.79s/it][A
     98%|█████████▊| 1242/1261 [32:51<00:33,  1.74s/it][A
     99%|█████████▊| 1243/1261 [32:53<00:31,  1.76s/it][A
     99%|█████████▊| 1244/1261 [32:55<00:29,  1.76s/it][A
     99%|█████████▊| 1245/1261 [32:57<00:28,  1.76s/it][A
     99%|█████████▉| 1246/1261 [32:58<00:26,  1.74s/it][A
     99%|█████████▉| 1247/1261 [33:00<00:24,  1.73s/it][A
     99%|█████████▉| 1248/1261 [33:02<00:22,  1.71s/it][A
     99%|█████████▉| 1249/1261 [33:03<00:20,  1.71s/it][A
     99%|█████████▉| 1250/1261 [33:05<00:18,  1.68s/it][A
     99%|█████████▉| 1251/1261 [33:07<00:16,  1.66s/it][A
     99%|█████████▉| 1252/1261 [33:08<00:14,  1.64s/it][A
     99%|█████████▉| 1253/1261 [33:10<00:13,  1.63s/it][A
     99%|█████████▉| 1254/1261 [33:11<00:11,  1.62s/it][A
    100%|█████████▉| 1255/1261 [33:13<00:09,  1.61s/it][A
    100%|█████████▉| 1256/1261 [33:15<00:08,  1.63s/it][A
    100%|█████████▉| 1257/1261 [33:16<00:06,  1.62s/it][A
    100%|█████████▉| 1258/1261 [33:18<00:04,  1.63s/it][A
    100%|█████████▉| 1259/1261 [33:20<00:03,  1.63s/it][A
    100%|█████████▉| 1260/1261 [33:21<00:01,  1.64s/it][A
    [A

    [MoviePy] Done.
    [MoviePy] >>>> Video ready: challange_output.mp4 
    



```python
from IPython.display import HTML
HTML("""
<video width="640" height="360" controls>
  <source src="{0}">
</video>
""".format('challange_output.mp4'))
```





<video width="640" height="360" controls>
  <source src="challange_output.mp4">
</video>




## Pipeline (video)
### 1. Does the pipeline established with the test images work to process the video?

Yes it does! The final results are shown in the video above.

## README

### 1. Has a README file been included that describes in detail the steps taken to construct the pipeline, techniques used, areas where improvements could be made?

In this project we are using colors and edges for finding the lanes. I have testes all color spaces available in CV2 and also tested several edge detection methods with many different threshold to determine the best method for lane detection in the road. It was challenging to find the best combination of settings and method to use i order to detect the lane lines as accurately as possible. The challenge test scenarios provided by Udacity were very challenging for my algorithm and improvements needed to improve its performance. The best achieved time for processing each frame on an i7 1.7 Ghz cpu was 1.1 seconds and I can improve the processing time by using GPUs instead of a 4 core cpu.  
Overall this project was fun and easy but required a lot of testing for finding the best line detection strategy. In the future I would implement deep learning and motion estimation for improving the lane line detection.   

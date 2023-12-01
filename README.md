# Computer Vision

## Image Processing Functions
This repository contains Python implementations of essential image processing functions. The provided functions operate on two-dimensional arrays representing images and are designed to handle various tasks such as convolution, color to grayscale conversion, image noise addition, image filtering, and edge detection.

## Functions
<ol>
  <li>myConv2(A, B, 'param')</li>
  Convolution function that takes two two-dimensional arrays (images) A and B as input and returns the result of the convolution operation between them. The dimensions of the output are determined based on the definition of the convolution operation.
  
  <li>myColorToGray(A, 'param')</li>
  Converts a color image represented by array A into its black and white version.
  
  <li>myImNoise(A, 'param')</li>
  Adds noise to the input image represented by array A. The type of noise ('gaussian' or 'saltandpepper') is specified by the user.
  
  <li>myImFilter(A, 'param')</li>
  Filters the input image represented by array A. The filtering can be performed using either the 'mean' or 'median' method. The user can set the size of the filters, and the program validates the input for filter size.

  <li>myEdgeDetection(A, 'param')</li>
  Detects edges in the black and white image represented by array A. Three edge detection methods are available: Sobel, Prewitt, and Laplacian.
</ol>

## Demo Script (demo1.py)
The repository includes a demonstration script, demo1.py, showcasing the usage of the implemented functions. The script performs the following operations:
<ol>
  <li>Reads an image from disk and converts it to black and white.</li>
  <li>Introduces noise into the image using myImNoise.</li>
  <li>De-noises the image using myImFilter.</li>
  <li>Prints the processed image at each stage.</li>
  <li>Computes the edges of the original image using myEdgeDetection.</li>
  <li>Constructs filtered images using different kernel sizes and calculates edges for each.</li>
  <li>Repeats the above steps for three edge detection methods.</li>
</ol>

## Note
The implementation avoids the use of ready-made libraries for convolution, edge detection, applying filters, and adding noise.


Feel free to explore the repository, use the functions, and incorporate them into your image processing projects!

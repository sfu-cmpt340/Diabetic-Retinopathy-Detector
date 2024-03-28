#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import numpy as np

def detect_lesions(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the detected contours
    mask = np.zeros_like(gray)

    # Draw the contours on the mask
    cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

    # Apply the mask to the original image
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the original image, detected contours, and segmented image
    cv2.imshow('Original Image', image)
    cv2.imshow('gray Image', gray)
    cv2.imshow('blurred Image', blurred)
    cv2.imshow('mask', mask)
    cv2.imshow('Detected Contours', cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2))
    cv2.imshow('Segmented Image', segmented_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image_path = 'C:/Users/ASHISH/OneDrive/Desktop/testimage.jpg'
detect_lesions(image_path)
print("Hello sidharth Sharma ji")


# In[ ]:





# In[ ]:





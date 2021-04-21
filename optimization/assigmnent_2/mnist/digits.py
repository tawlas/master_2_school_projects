import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_digits(image, digits, rects):
    image = image.copy()
    for rect in rects:
        cv2.rectangle(image, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
    plt.imshow(image)
    plt.show()
    n = int(np.ceil(np.sqrt(len(digits))))
    for i, roi in enumerate(digits):
        plt.subplot(n, n, i+1)
        plt.imshow(roi, cmap='gray')
        plt.axis('off')
    plt.show()
    

def find_digits(image):
    
    # Convert the image to grayscale and apply Gaussian filtering
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Threshold the image
    _, black_white_image = cv2.threshold(gray_image, 90, 255, cv2.THRESH_BINARY_INV)

    # Find contours in the image
    ctrs = find_contours(black_white_image)    
    
    # Exctract digits from the image
    digits, rects = extract_digits(black_white_image, ctrs)

    return digits, rects
    

def extract_digits(image, ctrs):  
    rects = []
    digits = []
    for ctr in ctrs:
        rect = cv2.boundingRect(ctr)
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = image[pt1:pt1+leng, pt2:pt2+leng]
        if roi.shape[0] < 28 or roi.shape[1] < 28:
            continue
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        digits.append(roi)
        rects.append(rect)
    digits = np.array(digits)
    return digits, rects


def find_contours(image):
    outs = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(outs) == 2:
        return outs[0]
    elif len(outs) == 3:
        return outs[1]
    else:
        raise Exception("OpenCV changed their cv2.findContours return signature yet again.")
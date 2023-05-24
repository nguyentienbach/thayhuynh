import cv2
import pytesseract
import re
import numpy as np
pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract"
# Load an image
img = cv2.imread('cmnd3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
boxes = pytesseract.image_to_data(img)
print(boxes)
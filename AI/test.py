import cv2
import pytesseract
import PIL
from matplotlib import pyplot as plt
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd="C:\\Program Files\\Tesseract-OCR\\tesseract"
tessdata_dir_config = r'--tessdata-dir "C:\\Program Files\\Tesseract-OCR\\tessdata" -l vie'

img_path="cmnd3.jpg"
image = cv2.imread(img_path)
image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

plt.figure(figsize=(8, 15))
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
image = cv2.threshold(image, 125, 255, cv2.THRESH_BINARY)[1]
image = cv2.medianBlur(image, 3)

plt.figure(figsize=(8, 15))
plt.imshow(image, cmap='gray', vmin=0, vmax=255)
plt.show()
# hImg,wImg,_ = image.shape
# boxes = pytesseract.image_to_boxes(image)
# for b in boxes.splitlines():
#    # print(b)
#     b= b.split(' ')
#     #print(b)
#     x,y,w,h = int(b[1]),int(b[2]),int(b[3]),int(b[4])
#     cv2.rectangle(image,(x,hImg-y),(w,hImg-h),(0,0,255),1)
#     cv2.putText(image,b[0],(x,hImg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(50,50,255),1)
# cv2.imshow('Result',image)
# cv2.waitKey(0)
# text = pytesseract.image_to_string(image, lang='vie')
# viegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through the contours and find the largest one, which is the ID card
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the ID card contour
mask = np.zeros(image.shape[:2], dtype=np.uint8)
cv2.drawContours(mask, [largest_contour], 0, 255, -1)

# Apply the mask to the image to extract the ID card
id_card = cv2.bitwise_and(image, image, mask=mask)

# Apply OCR to the ID card
text = pytesseract.image_to_string(id_card, lang='vie', config='--psm 6')

# Use regular expression to extract the ID card number
id_number = re.findall(r'\d{9}', text)

# Print the extracted ID card number
print("ID card number: ", id_number[0])

# Write the ID card number to a text file
with open('id_number.txt', 'w') as f:
    f.write(id_number[0])
boxes = pytesseract.image_to_data(image, lang='vie')
print(boxes)
for x,b in enumerate(boxes.splitlines()):
    if x!=0:
        b = b.split()
        if len(b)==12:
            # print(b)
            x,y,w,h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
            cv2.rectangle(image, (x,y), (x+w, h+y), (0,0,255),1)
            cv2.putText(image,b[11],(x,y),cv2.FONT_HERSHEY_PLAIN,1,(50,50,255),1)
cv2.imshow('result', image)
cv2.waitKey()
# cv2.imshow('ID card', id_card)
# cv2.waitKey()
# with open("doc.txt", "a", encoding="utf-8") as f:
#     f.writelines(boxes)
#pan_num = re.compile(r'SoCCCD\n.+')
#pan_no = re.search(pan_num, text)
#pan_id = ''
#if pan_no:
 #   pan_id = pan_no.group(0).split('\n')[-1]
  #  print(pan_id)
# data = re.compile(r'\n.*\n.*.\n\d{2}/\d{4}')
# data = re.search(data, text).group(0)
# #print(re.search('[A-Z].*', data).group(0))
# #print(data)
# #rint('SoCCCD:', pan_id)
# #print('HoVaTen:', data.split('\n')[1])9
# #print('DOB:', data.split('\n')[2])
#
# match = re.search(data, text)
# if match:
#     data = match.group(0)
import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread(r'C:\Users\hp\Documents\MATLAB\Datasets\Patient_1.jpg')
# create a mask image of the same shape as input image, filled with 0s (black color)
mask = np.zeros_like(image)
rows, cols,_ = mask.shape
# create a white filled ellipse
mask=cv2.ellipse(mask, center=(110, 176), axes=(35, 15), angle=0, startAngle=0, endAngle=360, color=(255,255,255), thickness=-1)
# Bitwise AND operation to black out regions outside the mask
result = np.bitwise_and(image,mask)
# Convert from BGR to RGB for displaying correctly in matplotlib
# Note that you needn't do this for displaying using OpenCV's imshow()
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
# Plotting the results
plt.subplot(131)
plt.imshow(image_rgb);
plt.subplot(132)
#plt.imshow(mask_rgb)
#plt.subplot(133)
plt.imshow(result_rgb);
plt.show()
rgb_img = cv2.cvtColor(result_rgb, cv2.COLOR_BGR2RGB) #cropped image is stored in result_rgb
cv2.imshow("Img",rgb_img)
cv2.waitKey(0)

red_pixels = rgb_img[:,:,2];
green_pixels = rgb_img[:,:,1];
blue_pixels = rgb_img[:,:,0];

#Performing operations on red pixels of the extracted cropped image.
red_pixels_1 = red_pixels.reshape(-1)
red_pixels_count = len(red_pixels_1)


for i_1 in range (1,red_pixels_count):
    red_pixels_2 = sum(red_pixels_1);

print(red_pixels_2);
count_red = 0;

for i_2 in range(1,red_pixels_count):
    if(red_pixels_1[i_2]>0):
        count_red = count_red+1;

mean_red_pixel_intensity= red_pixels_2/count_red;

#Performing operations on green pixels of the extracted cropped image.

green_pixels_1 = green_pixels.reshape(-1)
green_pixels_count = len(green_pixels_1)


for i_1 in range (1,green_pixels_count):
    green_pixels_2 = sum(green_pixels_1);

print(green_pixels_2);

count_green = 0;

for i_2 in range(1,green_pixels_count):
    if(green_pixels_1[i_2]>0):
        count_green = count_green+1;

mean_green_pixel_intensity= green_pixels_2/count_green;

diff_pixels = mean_red_pixel_intensity-mean_green_pixel_intensity;


print('Status of Patient :');
if (diff_pixels > 59):
    print("Non-Anaemic");
else:
    print("Anaemic");

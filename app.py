import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.preprocessing import image
import os
from werkzeug.utils import secure_filename
st.set_option('deprecation.showfileUploaderEncoding', False)
from keras.models import load_model
import cv2 as cv
html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:50px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:40px;color:white;margin-top:10px;">Digital Image Processing lab</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
  
st.title("""
       Transformation and Rotation using GUI 
         """
         )
file= st.file_uploader("Please upload image", type=("jpg", "png"))


import cv2
from  PIL import Image, ImageOps
def import_and_predict(image_data, angle):
     
  if angle == "30":
    img_RO = cv.imread("img1.PNG")
# convert from BGR to RGB so we can plot using matplotlib
    image4 = cv.cvtColor(img_RO, cv.COLOR_BGR2RGB)
# disable x & y axis
    plt.axis('off')
# show the image
    plt.imshow(image4)
    plt.show()
# get the image shape
    rows, cols, dim = image4.shape
#angle from degree to radian
    angle = np.radians(30)
#transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                	[np.sin(angle), np.cos(angle), 0],
                	[0, 0, 1]])
# apply a perspective transformation to the image
    rotated_img = cv.warpPerspective(image4, M, (int(cols),int(rows)))
# disable x & y axis
#plt.axis('off')
# show the resulting image
    plt.imshow(rotated_img)
    plt.show()
    st.image(rotated_img, caption='rotated image', use_column_width=True)
# save the resulting image to disk
    plt.imsave("/spy.png", rotated_img)

  elif angle == "45":
    img_RO = cv.imread("img1.PNG")
# convert from BGR to RGB so we can plot using matplotlib
    image4 = cv.cvtColor(img_RO, cv.COLOR_BGR2RGB)
# disable x & y axis
    plt.axis('off')
# show the image
    plt.imshow(image4)
    plt.show()
# get the image shape
    rows, cols, dim = image4.shape
#angle from degree to radian
    angle = np.radians(45)
#transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                	[np.sin(angle), np.cos(angle), 0],
                	[0, 0, 1]])
# apply a perspective transformation to the image
    rotated_img = cv.warpPerspective(image4, M, (int(cols),int(rows)))
# disable x & y axis
#plt.axis('off')
# show the resulting image
    plt.imshow(rotated_img)
    plt.show()
    st.image(rotated_img, caption='rotated image', use_column_width=True)
# save the resulting image to disk
    plt.imsave("/spy.png", rotated_img)

  elif angle == "60":
    img_RO = cv.imread("img1.PNG")
# convert from BGR to RGB so we can plot using matplotlib
    image4 = cv.cvtColor(img_RO, cv.COLOR_BGR2RGB)
# disable x & y axis
    plt.axis('off')
# show the image
    plt.imshow(image4)
    plt.show()
# get the image shape
    rows, cols, dim = image4.shape
#angle from degree to radian
    angle = np.radians(60)
#transformation matrix for Rotation
    M = np.float32([[np.cos(angle), -(np.sin(angle)), 0],
                	[np.sin(angle), np.cos(angle), 0],
                	[0, 0, 1]])
# apply a perspective transformation to the image
    rotated_img = cv.warpPerspective(image4, M, (int(cols),int(rows)))
# disable x & y axis
#plt.axis('off')
# show the resulting image
    plt.imshow(rotated_img)
    plt.show()
    st.image(rotated_img, caption='rotated image', use_column_width=True)
# save the resulting image to disk
    plt.imsave("/spy.png", rotated_img)


  else:
    print("choose any direction, not found your requirement")
        
 
  return 0

if file is None:
  st.text("Please upload an Image file")
else:
  file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(file,caption='Uploaded Image.', use_column_width=True)
    
if st.button("Display image"):
  angle = st.selectbox('angle', ('30', '45','60'))
  st.write('You selected: ', angle)
  result=import_and_predict(image, angle)
  
if st.button("About"):
  st.header(" YATIN AGARWAL ")
  st.subheader("Student, Department of Computer Engineering")
html_temp = """
   <div class="" style="background-color:orange;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:20px;color:white;margin-top:10px;">Digital Image processing Experiment</p></center> 
   </div>
   </div>
   </div>
   """
st.markdown(html_temp,unsafe_allow_html=True)
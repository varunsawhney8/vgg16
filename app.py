from PIL import Image
import tensorflow.keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import cv2
import streamlit as st
st.title("Image Classification Using VGG16")
st.write("This app classifies images based on VGG16 pretrained model.")
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img1=img
    #resize the image to 224x224 square shape
    img = img.resize((224,224))
    #convert the image to array
    img_array = img_to_array(img)
    #convert the image into a 4 dimensional Tensor
    #convert from (height, width, channels), (batchsize, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    #preprocess the input image array
    img_array = imagenet_utils.preprocess_input(img_array)
    #Load the model from internet / computer
    #approximately 530 MB
    pretrained_model = VGG16(weights="imagenet")
    #predict using predict() method
    prediction = pretrained_model.predict(img_array)
    #decode the prediction
    actual_prediction = imagenet_utils.decode_predictions(prediction)
    
    st.write("UPLOADED IMAGE")
    st.image(img1, caption='Uploaded Image.', use_column_width=True)
    st.write("predicted object is:")
    st.write(actual_prediction[0][0][1])
    st.write("with accuracy")
    st.write(actual_prediction[0][0][2]*100)

else:
    st.write("You haven't uploaded any image yet")
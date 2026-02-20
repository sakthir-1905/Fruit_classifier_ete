# Importing libraries

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

img_width, img_height = 150,150

model = load_model("model.h5")

st.set_page_config("Fruit Classifier")
st.title("Fruit Classifier")
st.write("Upload your file")

img = st.file_uploader("Upload your image")
if img != None:
    img = image.load_img(img, target_size = (img_width,img_height))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis = 0)
    img_array = img_array / 255.0

    pred = model.predict(img_array)

    classes = ['Apple', 'Banana','Grape', 'Mango','Strawberry']
    index = np.argmax(pred)
    st.info(classes[index])




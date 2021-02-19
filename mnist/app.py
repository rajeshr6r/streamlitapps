import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas


#useful variables 
global loaded_model

#Wide layout
st.set_page_config(layout="wide")

#page title
st.title('Handwritten Digit Recognizer')
st.markdown('''
Try to write a digit!
''')

#load model from the model directory
@st.cache
def load_model():
    try:
        pass
        st.write(f"Model Loaded Successfully")
    except:
        st.write(f"Model Load Error")

    
model_load_state = st.text('Loading Model...')
loaded_model=load_model() # load the model here 
model_load_state.text("Model Loaded ! (using st.cache)")


#set canvas
SIZE = 392
mode = st.checkbox("To draw click the checkbox and use your mouse or touch pad", False)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')


if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = loaded_model.predict(test_x.reshape(1, 28, 28))
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
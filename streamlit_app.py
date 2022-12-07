from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
import keras
from keras.models import *
from PIL import Image
import keras.utils as image_postproccess
import numpy as np
import imageio
import os
import zipfile
import tempfile

stream = st.file_uploader('Choose a model', type='zip')
if stream is not None:
  myzipfile = zipfile.ZipFile(stream)
  with tempfile.TemporaryDirectory() as tmp_dir:
    myzipfile.extractall(tmp_dir)
    root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
    model_dir = os.path.join(tmp_dir, root_folder)
    model = keras.models.load_model(model_dir)


st.write((os.getcwd()))
st.write((os.listdir()))

# """
# # Welcome to Streamlit!
#
# Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:
#
# If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
# forums](https://discuss.streamlit.io).
#
# In the meantime, below is an example of what you can do with just a few lines of code:
# """

"""
Upload Patient Xray below
"""


# model = load_model("models_deploy/Detect_Covid-2022-11-28--02-30-48.h5")
# @st.cache

# with st.echo(code_location='below'):
#     total_points = st.slider("Number of points in spiral", 1, 5000, 2000)
#     num_turns = st.slider("Number of turns in spiral", 1, 100, 9)
#
#     Point = namedtuple('Point', 'x y')
#     data = []
#
#     points_per_turn = total_points / num_turns
#
#     for curr_point_num in range(total_points):
#         curr_turn, i = divmod(curr_point_num, points_per_turn)
#         angle = (curr_turn + 1) * 2 * math.pi * i / points_per_turn
#         radius = curr_point_num / total_points
#         x = radius * math.cos(angle)
#         y = radius * math.sin(angle)
#         data.append(Point(x, y))
#
#     st.altair_chart(alt.Chart(pd.DataFrame(data), height=500, width=500)
#         .mark_circle(color='#0068c9', opacity=0.5)
#         .encode(x='x:Q', y='y:Q'))
# model_file = st.file_uploader("Choose a Model")
uploaded_file = st.file_uploader("Choose a file")
if (uploaded_file is not None) and (model is not None):
    # model = load_model(model_file)
    classes = {'covid': 0, 'normal': 1, 'pneumonia_bacteria': 2, 'pneumonia_virus': 3}
    inv_classes = {v: k for k, v in classes.items()}

    # img_path = "/Users/ryanwest/OMSCS/cs6440/NORMAL2-IM-0372-0001.jpeg"
    image = Image.open(uploaded_file).convert('RGB')#img_path
    target_size = 227
    # img_keras = image_postproccess.load_img(img_path, target_size=(target_size, target_size))
    image_resized = image.resize((target_size, target_size))
    image_array = image_postproccess.img_to_array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255
    predict_x = model.predict(image_array, verbose=False)
    y_pred = np.argmax(predict_x[0])
    y_pred_proba = max(predict_x[0])
    st.title(f"Patient XYZ Diagnosis: {inv_classes[y_pred].upper()} xray")
    st.header(f"{str(round(y_pred_proba,4))} confidence")

    st.image(image, caption=f"Patient XYZ Diagnosis: {inv_classes[y_pred].upper()} xray \n {y_pred_proba} confidence")

# import urllib.request
# @st.experimental_singleton
# def load_model():
#     if not os.path.isfile('model.h5'):
#         urllib.request.urlretrieve('https://github.gatech.edu/rwest61/xray-classification/blob/master/models_deploy/Detect_Covid-2022-11-28--02-30-48.h5', 'model2.h5')
#     return keras.models.load_model('model.h5')
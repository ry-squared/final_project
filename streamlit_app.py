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

"""
Upload Model Below
"""

stream = st.file_uploader('Choose a model', type='zip')
@st.experimental_singleton
def load_model_from_zip():
    if stream is not None:
      myzipfile = zipfile.ZipFile(stream)
      with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()[0] # e.g. "model.h5py"
        model_dir = os.path.join(tmp_dir, root_folder)
        model = keras.models.load_model(model_dir)
    return model
model = load_model_from_zip()

if model is not None:
    st.subheader("Model already loading.. skip model upload")

"""
Upload Patient Xray below
"""

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

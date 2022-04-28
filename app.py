import streamlit as st
import degirum as dg
from PIL import Image

zoo=dg.connect_model_zoo()
model=zoo.load_model('mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1')
uploaded_file=st.file_uploader('input image')

image = Image.open(uploaded_file)

st.image(uploaded_file, caption='input image')
predictions=model(image)
st.image(predictions.image_overlay,caption='bounding boxes')

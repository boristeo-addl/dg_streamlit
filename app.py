import streamlit as st
import degirum as dg
from PIL import Image

degirum_public_zoo="https://api.github.com/repos/degirum/public_model_zoo/releases/latest"
zoo=dg.connect_model_zoo(zoo_url=degirum_public_zoo,token=st.secrets["GH_TOKEN"])

st.title('DeGirum Public Model Zoo Demo for CPU')

st.header('Specify Model Options Below')
precision=st.radio("Choose model precision",("Float","Quant","Don't Care"),index=2)
runtime_agent=st.radio("Choose runtime agent",("TFLite","N2X","Don't Care"),index=2)
precision=precision if precision!="Don't Care" else ""
runtime_agent=runtime_agent if runtime_agent!="Don't Care" else "" 
model_options=zoo.list_models(device='CPU',precision=precision,runtime=runtime_agent)
st.header('Choose and Run a Model')
st.text('Select a model and upload an image. Then click on the submit button')
with st.form("model_form"):
    model_name=st.selectbox("Choose a Model from the list", model_options)
    uploaded_file=st.file_uploader('input image')
    submitted = st.form_submit_button("Submit")
    if submitted:
        model=zoo.load_model(model_name)
        st.write("Model loaded successfully")
        image = Image.open(uploaded_file)
        predictions=model(image)
        st.image(predictions.image_overlay,caption='bounding boxes')

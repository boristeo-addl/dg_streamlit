import streamlit as st
import degirum as dg
from PIL import Image

zoo=dg.connect_model_zoo(token=st.secrets["GH_TOKEN"])
model=None#zoo.load_model('mobilenet_v2_ssd_coco--300x300_quant_n2x_cpu_1')
hw_option = st.radio("Choose target HW", ("ORCA", 'CPU', "EDGETPU", "Don't care"))
precision=st.radio("Choose model precision",("Float","Quant","Don't Care"))
runtime_agent=st.radio("Choose runtime agent",("TFLite","N2X","Don't Care"))
pruned=st.radio("Choose model density",("Dense","Pruned","Don't Care"))

hw_option=hw_option if hw_option!="Don't Care" else "" 
precision=precision if precision!="Don't Care" else ""
pruned=pruned if pruned!="Don't Care" else ""
runtime_agent=runtime_agent if runtime_agent!="Don't Care" else "" 
model_options=zoo.list_models(device=hw_option,precision=precision,runtime=runtime_agent,pruned=pruned)
st.write('List of supported models')
st.write(model_options)
with st.form("model_form"):
    model_name=st.text_input("Model Name", value="")
    uploaded_file=st.file_uploader('input image')
    submitted = st.form_submit_button("Submit")
    if submitted:
        model=zoo.load_model(model_name)
        st.write("Model loaded successfully")
        image = Image.open(uploaded_file)
        predictions=model(image)
        st.image(predictions.image_overlay,caption='bounding boxes')

import streamlit as st
import degirum as dg
from PIL import Image

zoo=dg.connect_model_zoo('https://cs.degirum.com/degirum_com/yolov5_cpu_nota',token=st.secrets["DG_TOKEN"])
 
st.title('DeGirum Cloud Platform Demo of YOLOv5 Models')
model_options=zoo.list_models()
st.header('Choose and Run a Model')
st.text('Select a model and upload an image. Then click on the submit button')
with st.form("model_form"):
    model_name=st.selectbox("Choose a Model from the list", model_options)
    uploaded_file=st.file_uploader('input image')
    submitted = st.form_submit_button("Submit")
    if submitted:
        model=zoo.load_model(model_name)
        model.overlay_font_scale=3
        model.overlay_line_width=6
        st.write("Model loaded successfully")
        image = Image.open(uploaded_file)
        predictions=model(image)
        st.image(predictions.image_overlay,caption='Image with Bounding Boxes')
            

import streamlit as st
import degirum as dg
from PIL import Image

zoo=dg.connect('https://cs.degirum.com/degirum_com/public',token=st.secrets["DG_TOKEN"])
lp_det_model = zoo.load_model("yolo_v5s_lp_det--512x512_quant_n2x_orca_1")
ocr_model = zoo.load_model("yolo_v5s_lp_ocr--256x256_quant_n2x_orca_1")

# adjust some model properties
lp_det_model.output_confidence_threshold = 0.7
lp_det_model.overlay_font_scale = 1.5
lp_det_model.overlay_alpha = 1
ocr_model.overlay_show_labels = True
ocr_model.overlay_font_scale = 1.5
ocr_model.overlay_alpha = 1
ocr_model.output_top_k = 1
 
st.title('Demo of License Plate Recognition using the DeGirum Cloud Platform')
st.text('Upload an image. Then click on the submit button')
with st.form("model_form"):
    uploaded_file=st.file_uploader('input image')
    submitted = st.form_submit_button("Submit")
    if submitted:
        license_plates = lp_det_model(Image.open(uploaded_file))
        with ocr_model: # performance optimization to keep connection to mask_det_model open
            for lp in license_plates.results:
                lp_box = lp.image.crop(face['bbox'])
                digits = ocr_model(lp_box)
                lp["label"] = digits.results[0]["label"]
        st.image(license_plates.image_overlay,caption='Image with License Plate Information')
            

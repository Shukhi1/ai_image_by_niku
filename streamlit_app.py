
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from streamlit_lottie import st_lottie
import requests
from io import BytesIO
import base64

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

model = tf.keras.models.load_model("detector_model.h5")
IMG_SIZE = (128, 128)


st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="wide",
)

lottie_animation = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_jcikwtux.json")
st_lottie(lottie_animation, speed=1, height=200, key="ai")

st.markdown("<h1 style='text-align: center; color: #4B0082;'>AI Image Detector</h1>", unsafe_allow_html=True)
st.markdown("---")


st.markdown(
    """
    <div style='background-color:#f0f2f6;padding:20px;border-radius:15px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.1);'>
        <h3 style='text-align:center;color:#333;'>Upload an image to check if it's AI-generated or real</h3>
    </div>
    """,
    unsafe_allow_html=True
)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file:
    image = Image.open(uploaded_file)

    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    # Display image in a fixed-size card without cutting (scaled down)
    st.markdown(
        f"""
        <div style='width:500px; height:400px; margin:auto; border-radius:15px;
                    overflow:hidden; box-shadow:0px 4px 10px rgba(0,0,0,0.1); display:flex; align-items:center; justify-content:center;'>
            <img src="data:image/png;base64,{img_str}" 
                 style='max-width:100%; max-height:100%; object-fit:contain;'/>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Preprocess for model
    img = image.resize(IMG_SIZE)
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        label = "Ai generated Image"
        color = "#28a745"
        confidence = prediction * 100
    else:
        label = "Real Image"
        color = "#dc3545"
        confidence = (1 - prediction) * 100

    # Display label and confidence
    st.markdown(f"<h2 style='text-align: center; color: {color};'>{label}</h2>", unsafe_allow_html=True)
    st.progress(int(confidence))
    st.markdown(
        f"<p style='text-align:center;font-size:18px;color:#white;'><b>Confidence:</b> {round(confidence,2)}%</p>",
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div style='background-color:#fff3cd;padding:15px;border-left:5px solid #ffeeba;
                    border-radius:10px;margin-top:10px;color:#856404;'>
        ⚠️ Prediction is based on AI model; results may vary.
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#003366; font-weight:bold;'>Made with ❤️ using Streamlit, TensorFlow & modern design</p>",
    unsafe_allow_html=True
)
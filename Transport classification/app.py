# Kerakli kutubxonalar
import plotly
import fastai
import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Title
st.title('Transportni klassifikatsiya qiluvchi model')

# Rasmni joylash
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'gif', 'svg', 'jpg'])
if file:
    st.image(file)
    # PIL orqali konvert qilamiz
    img = PILImage.create(file)

    # Model
    model = load_learner('transport_model.pkl')

    # Prediction
    pred, pred_id, probs = model.predict(img)

    st.success(f'Bashorat: {pred}')
    st.info(f'Ehtimollik: {probs[pred_id] * 100:.2f} %')

    # Ploting
    fig = px.bar(x=probs * 100, y=model.dls.vocab)
    st.plotly_chart(fig)

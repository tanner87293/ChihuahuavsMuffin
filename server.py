import streamlit as st
import urllib.request
from fastai.vision.all import *




model = load_learner('ChihuahuaVsMuffin.pkl')


def predict(image):
    img = PILImage.create(image)  # Use PILImage.create to open the image
    pred_class, pred_idx, outputs = model.predict(img)
    likelihood_is_chihuahua = outputs[1].item()
    if likelihood_is_chihuahua > 0.9:
        return "Chihuahua"
    elif likelihood_is_chihuahua < 0.1:
        return "Muffin"
    else:
        return "Not sure... try another picture!"


st.title("Chihuahua vs. Muffin Classifier by Leon Chen")
st.write("Upload an image, and I'll tell you whether it's a Chihuahua or a Muffin!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        prediction = predict(uploaded_file)
        st.write(prediction)

st.text("Built with Streamlit and Fastai")

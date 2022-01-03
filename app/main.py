import streamlit as st 
from PIL import Image
from classify import predict
import pandas as pd
import numpy as np


st.title("Skin Lession Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predictions = predict(image)
    tableArea = st.empty()
    tableArea = tableArea.table()
    with st.spinner("Prediction In Progress"):
        for ix, (key, val) in enumerate(predictions):
            tableArea.add_rows(pd.DataFrame({"Lession Type": key, "Probability %": "%{:.4f}".format(val) }, index = [ix]))
        st.success("Success")

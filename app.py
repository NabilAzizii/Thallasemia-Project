import streamlit as st
from PIL import Image
import numpy as np
import json
import time
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.densenet import preprocess_input

model = load_model("model_thalassemia_classifier_100.h5")
with open("class.json") as fin:
    tag2class = json.load(fin)
    class2tag = {v: k for k, v in tag2class.items()}

class ClassifyModel:
    def __init__(self):
        self.model = None
        self.class2tag = None
        self.tag2class = None

    def load(self, model_path="model_thalassemia_classifier_100.h5", class_path="class.json"):
        self.model = load_model(model_path)
        with open(class_path) as fin:
            self.tag2class = json.load(fin)
            self.class2tag = {v: k for k, v in self.tag2class.items()}

    def predict(self, image_array):
        pred = self.model.predict(image_array)
        pred_digits = np.argmax(pred, axis=1)

        return pred_digits, pred

m = ClassifyModel()
m.load()

st.title('Thalassemia Classification')

st.header("Upload a blood image")

st.sidebar.title("About")

st.sidebar.info(
    "An application that utilizes artificial intelligence to accurately classify thalassemia disease through blood image analysis. Using advanced machine learning models, we are committed to providing innovative solutions that aid early diagnosis and effective management of this disease.")


st.sidebar.title("Creator")

st.sidebar.info(
    "Fikri Dwi Alpian - 120450022"
)
st.sidebar.info(
    "Anastasya Nurfitriyani Hidayat - 120450080	"
)
st.sidebar.info(
    "Muhammad Nabil Azizi - 120450090"
)

st.sidebar.title("Source")
st.sidebar.info("[Link to Original Repository](https://github.com/sains-data/Thalassemia-Classification-CNN.git)")

uploaded_file = st.file_uploader("Choose a blood image...", type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
    img = keras_image.load_img(uploaded_file, target_size=(100, 100))
    img_array = keras_image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    st.image(img, caption=f'Uploaded Image: {uploaded_file.name}')

    if st.button('Predict'):
        start_time = time.time()
        pred_digits, pred_probabilities = m.predict(img_array)
        end_time = time.time()
        if len(pred_digits) > 0 and pred_digits[0] < len(class2tag):
            predicted_label = class2tag[pred_digits[0]]
            confidence = pred_probabilities[0][pred_digits[0]]
            st.write(f"Predicted Label: **{predicted_label}**")
            st.write(f"Confidence: **{round(float(confidence), 4) * 100}%**")
        else:
            st.write("Error: Predicted class index out of bounds.")
        execution_time = end_time - start_time
        st.write(f"Time taken for prediction: **{execution_time:.4f} seconds**")
        



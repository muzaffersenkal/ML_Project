from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv()

MODEL_PATH = os.getenv('MODEL_PATH')
labels = ["Actinic keratoses", "Basal cell carcinoma","Benign keratosis-like lesions", "Dermatofibroma", "Melanocytic nevi", "Melanoma", "Vascular lesions"]

def predict(image): 
    model = load_model(MODEL_PATH)
    image = image.resize((224,224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # scale image
    image /= 255.0 
    # reshape data for the model
    image = np.expand_dims(image, axis=0)
    # predict image
    prediction = model.predict(image)
    result = list(zip(labels,prediction[0]))
    # sort values
    result.sort(key=lambda x:x[1], reverse=True)
    return  result
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import visualkeras

model = load_model("data/weights/cnn/model_best.hdf5")

visualkeras.layered_view(model, max_xy=300, legend=True, to_file='graphs/cnn_architecture.png')
# Backend for the pneumonia detection app: loads the model and predicts one chest X-ray
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pydicom
import tensorflow as tf

# Folder with the saved model and its settings
MODEL_DIR = Path(__file__).parent / 'model'


# ---------------------------------------------------------------------
# 1) load_model_and_config: load the saved model and its settings
# ---------------------------------------------------------------------
def load_model_and_config():

    # Load the saved Keras model
    model = tf.keras.models.load_model(MODEL_DIR / 'best_pneumonia_model.keras')

    # Load the settings (threshold, image size and CLAHE settings)
    with open(MODEL_DIR / 'model_config.json') as f:
        config = json.load(f)

    return model, config


# ---------------------------------------------------------------------
# 2) read_image: read an uploaded DICOM, PNG or JPG file
# ---------------------------------------------------------------------
def read_image(file_bytes, filename):

    # DICOM file: read the pixel array
    if filename.lower().endswith('.dcm'):
        ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
        img = ds.pixel_array.astype('float32')

        # MONOCHROME1 stores black as the highest value, so invert it
        if getattr(ds, 'PhotometricInterpretation', '') == 'MONOCHROME1':
            img = img.max() - img
        return img

    # PNG or JPG file: decode the bytes into an image
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError('The file could not be read as an image.')
    return img


# ---------------------------------------------------------------------
# 3) preprocess: same preprocessing as the training images
# ---------------------------------------------------------------------
def preprocess(img, config):

    # Scale the pixel values to 0-255 (dtype=cv2.CV_8U rounds, as in the notebook, instead of cutting decimals)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Convert to grayscale if the image has colour channels
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resize to the image size used in training
    size = config['image_size']
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)

    # Improve the local contrast with CLAHE
    clahe = cv2.createCLAHE(clipLimit=config['clahe_clip_limit'],
                            tileGridSize=tuple(config['clahe_tile_grid']))
    img = clahe.apply(img)

    # Scale to 0-1 and add the batch and channel axes: shape (1, size, size, 1)
    x = img.astype('float32')[None, ..., None] / 255.0

    return x, img


# ---------------------------------------------------------------------
# 4) predict: pneumonia probability and predicted class
# ---------------------------------------------------------------------
def predict(model, config, x):

    # Probability of pneumonia (0 to 1)
    prob = float(model.predict(x, verbose=0).ravel()[0])

    # Predicted class at the tuned threshold
    thr = config['threshold']
    label = config['class_names'][1] if prob >= thr else config['class_names'][0]

    return prob, label, thr

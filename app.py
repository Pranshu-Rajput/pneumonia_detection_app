# Frontend: Streamlit app to upload a chest X-ray and see the predicted class and probability
import streamlit as st

# Backend functions from inference.py
from inference import load_model_and_config, read_image, preprocess, predict

# Page settings
st.set_page_config(page_title='Pneumonia Detection', layout='centered')


# Load the model once and keep it in memory between requests
@st.cache_resource
def get_model():
    return load_model_and_config()

model, config = get_model()


# ---------------------------------------------------------------------
# Page title and file upload
# ---------------------------------------------------------------------
st.title('Pneumonia Detection from Chest X-rays')
st.write('Upload a chest X-ray (DICOM, PNG or JPG). The model returns the predicted class '
         'and the probability of pneumonia.')

uploaded = st.file_uploader('Chest X-ray file', type=['dcm', 'png', 'jpg', 'jpeg'])


# ---------------------------------------------------------------------
# Prediction and results (shown after a file is uploaded)
# ---------------------------------------------------------------------
if uploaded is not None:

    # Read the file, preprocess it like the training images and predict
    try:
        img = read_image(uploaded.getvalue(), uploaded.name)
        x, shown = preprocess(img, config)
        prob, label, thr = predict(model, config, x)
    except Exception as e:
        st.error(f'Could not process this file: {e}')
        st.stop()

    # Show the image on the left and the results on the right
    col1, col2 = st.columns(2)

    with col1:
        st.image(shown, caption='Preprocessed image (224 x 224)', clamp=True, width=300)

    with col2:
        st.metric('Predicted class', label)
        st.metric('Probability of pneumonia', f'{prob:.1%}')
        st.metric('Probability of no pneumonia', f'{1 - prob:.1%}')
        st.progress(min(max(prob, 0.0), 1.0))
        st.caption(f'Decision threshold: {thr:.3f} (tuned for at least 85% recall on validation data). '
                   f'Model: {config["model_name"]}.')

    # Reminder that the result supports, and does not replace, a radiologist
    st.warning('Decision-support only: every result needs review by a radiologist.')

# Pneumonia Detection App

Streamlit app that predicts pneumonia from a chest X-ray (DICOM, PNG or JPG) and shows the
predicted class and probability. Decision-support only: results need radiologist review.

## Run with Docker (GitHub Codespaces or any machine with Docker)
```bash
docker build -t pneumonia-app .
docker run -p 8501:8501 pneumonia-app
```
Open port 8501 (in Codespaces: **Ports** tab -> forwarded address).

## Run without Docker
```bash
pip install -r requirements.txt
streamlit run app.py
```

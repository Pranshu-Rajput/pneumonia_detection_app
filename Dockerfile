# Container for the pneumonia detection app

# Start from a small Python image
FROM python:3.11-slim

# Install the system library that OpenCV needs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Work inside the /app folder
WORKDIR /app

# Install the Python libraries first (this step is reused between builds)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the app code and the saved model
COPY app.py inference.py ./
COPY model/ ./model/

# The app listens on port 8501
EXPOSE 8501

# Start the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

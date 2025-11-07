# Use an official Python 3.10 slim base image
FROM python:3.10-slim

# Install the system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy your app files into the container
COPY requirements.txt .
COPY app.py .

# Install your Python packages
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on (default 8501)
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
# Use an official Streamlit base image (which is Debian-based)
FROM streamlit/base:latest

# Install the system dependencies for OpenCV
# This is where we fix the error from your log
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy your app files into the container
WORKDIR /app
COPY requirements.txt .
COPY app.py .

# Install your Python packages
RUN pip install -r requirements.txt

# Expose the port Streamlit runs on
EXPOSE 8501

# The command to run your app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
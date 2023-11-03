# Use a base image with the required environment (e.g., Python, machine learning libraries)
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy your model training script and any necessary files to the container
COPY model_train.py /app/
COPY requirements.txt /app/
# COPY data/ /app/data/

# Install any dependencies if needed
RUN pip install -r requirements.txt

# Define a volume to save trained models on the host machine
VOLUME /app/models

# Command to execute when the container runs
CMD ["python", "model_train.py"]
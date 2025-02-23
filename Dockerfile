# Use an official lightweight Python base image
FROM python:3.10-slim

# Install dependencies and AWS CLI
RUN apt update -y && apt install awscli -y

# Set working directory inside the container
WORKDIR /app

# Copy all files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Command to run the FastAPI app
CMD ["python3", "app.py"]

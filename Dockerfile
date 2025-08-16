# Use official Python slim image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy required files
COPY requirements.txt /app/requirements.txt
COPY app.py /app/app.py
COPY models /app/models

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for Flask
EXPOSE 5000

# Start Flask app
CMD ["python", "app.py"]

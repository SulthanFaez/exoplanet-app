# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Command to run Streamlit app
CMD ["streamlit", "run", "testingapp.py", "--server.port=8501", "--server.address=0.0.0.0"]

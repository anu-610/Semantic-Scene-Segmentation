# Use a Python image with PyTorch pre-installed (saves time)
FROM datajoint/miniconda3:latest

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY web_interface/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONPATH=/app/Models:/app/Models/Ensemble

# Expose the HF port
EXPOSE 7860

# Run the app
CMD ["python", "web_interface/app.py"]
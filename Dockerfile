# 1. Use an official, lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies (if your ML libraries need them, e.g., gcc for compiling certain packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy only the requirements first (this optimizes Docker's caching mechanism)
COPY requirements.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of your project files into the container
COPY . .

# 7. Define the default command to run your pipeline
# REPLACE 'orchestration/main.py' with the actual script you use to run your pipeline
CMD ["python", "orchestration/run_pipeline.py"]
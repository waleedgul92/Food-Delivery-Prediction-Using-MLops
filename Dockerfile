# 1. Use an official Python slim image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Set helpful environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 4. Copy the requirements file
# Assuming reqs-docker.txt is the correct name from your log
COPY reqs-docker.txt .

# 5. Install the Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r reqs-docker.txt

# 6. Copy your application code
# This copies your 'app' folder, 'src' folder, etc.
COPY . .

# 7. Tell Docker the container will listen on port 8000
EXPOSE 8000

# 8. The command to run your application
CMD ["uvicorn", "app.Fast_api:app", "--host", "0.0.0.0", "--port", "8000"]

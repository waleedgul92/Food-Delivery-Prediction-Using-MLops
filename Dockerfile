FROM python:3.10-slim



WORKDIR /app




COPY app/ /app/app/
COPY reqs-docker.txt /app/reqs-docker.txt
COPY data/ /app/data/
COPY params.yaml /app/params.yaml
COPY models/ /app/models/
COPY artifacts/ /app/artifacts/
RUN  pip install -r reqs-docker.txt
COPY src/ /app/src/
COPY Fast_api.py /app/Fast_api.py
COPY src/mlflow_config.py /app/mlflow_config.py
COPY src/mlflow_config.py /app/src/mlflow_config.py
COPY log_data /app/log_data/

EXPOSE 5000


# 8. The command to run your application
CMD ["uvicorn", "app.Fast_api:app", "--host", "0.0.0.0", "--port", "8000"]
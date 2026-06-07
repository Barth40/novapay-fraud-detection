FROM python:3.11-slim

WORKDIR /app

COPY requirements-deploy.txt .

RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY app/ ./app/
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
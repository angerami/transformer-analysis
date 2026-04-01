FROM python:3.13.5-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY HF_requirements.txt ./
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r HF_requirements.txt

COPY *.py ./
COPY dashboards/ ./dashboards/

EXPOSE 7860
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health
ENTRYPOINT ["streamlit", "run", "dashboards/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]

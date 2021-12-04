FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="team-erc"

ENV WORKERS_PER_CORE=4 
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"

RUN mkdir /yolov5-fastapi

COPY requirements.txt /yolov5-fastapi

COPY . /yolov5-fastapi

WORKDIR /yolov5-fastapi

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
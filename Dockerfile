FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY models/model-2May-14th.pkl ./models/

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port 8000

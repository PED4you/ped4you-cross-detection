FROM fastai/fastai:latest

WORKDIR /usr/src/app

COPY requirements-fastapi.txt ./
RUN pip install -r requirements-fastapi.txt

COPY main.py ./
COPY models/model2.pkl ./models/

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port 8000

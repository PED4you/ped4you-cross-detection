FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY inference.py ./
COPY models/model2.pkl ./models/

EXPOSE 8000

CMD [ "python", "main.py"]

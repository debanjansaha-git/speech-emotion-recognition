FROM continuumio/miniconda3:latest

WORKDIR /app

RUN pip install -U pip --upgrade pip

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .
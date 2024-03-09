FROM apache/airflow:2.8.1

WORKDIR /opt/airflow
RUN pip install -U pip --upgrade pip
COPY . .
RUN pip install -r ./requirements.txt

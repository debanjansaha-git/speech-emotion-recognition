FROM apache/airflow:2.8.1

USER root
RUN apt-get update
RUN apt-get install -y libsndfile1

WORKDIR /opt/airflow/dags

COPY --chown=0 --chmod=777 ./dags /opt/airflow/dags
# COPY ./dags /opt/airflow/dags
COPY ./requirements.txt /opt/airflow/

USER ${AIRFLOW_UID:-0}
RUN pip install -U pip --upgrade pip
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION:-2.8.1}" -r /opt/airflow/requirements.txt
# RUN pip install -r /opt/airflow/requirements.txt
# RUN apt-get update
# RUN apt-get install libsndfile1
# RUN /usr/bin/python3 setup.py sdist
# RUN /usr/bin/python3 setup.py install --user
RUN pip install .
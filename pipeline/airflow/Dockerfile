FROM apache/airflow:2.8.1

WORKDIR /opt/airflow/dags

COPY --chown=0 --chmod=777 ./dags /opt/airflow/dags
COPY ./requirements.txt /opt/airflow/

RUN pip install -U pip --upgrade pip
RUN pip install -r /opt/airflow/requirements.txt
# RUN /usr/bin/python3 setup.py sdist
# RUN /usr/bin/python3 setup.py install --user
RUN pip install .
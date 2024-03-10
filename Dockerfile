FROM apache/airflow:2.8.1

WORKDIR /app

RUN pip install -U pip --upgrade pip

# TODO: Change this permission. 777 isnt safe
COPY --chown=0 --chmod=777 . /app

RUN pip install -r ./requirements.txt
RUN python setup.py sdist
RUN python setup.py install --user
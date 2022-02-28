FROM python:3.10.0

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Oslo

RUN pip install --upgrade pip
RUN useradd -m -s /bin/bash arturo_docker
USER arturo_docker
ENV PATH "$PATH:/home/arturo_docker/.local/bin"

USER root

COPY ./requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

ENV PYTHONPATH "${PYTHONPATH}:/home/arturo_docker/ml_churn_risk_forecast:"

USER arturo_docker
WORKDIR /home/arturo_docker/ml_churn_risk_forecast
ENTRYPOINT ["bash"]

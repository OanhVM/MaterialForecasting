FROM tensorflow/tensorflow:2.13.0-gpu

ENV APP_PATH /usr/src/app
ENV SRC_PATH ${APP_PATH}/src
ENV MODELS_PATH ${APP_PATH}/models

# Workaround for NVIDIA key issue https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1112667716
RUN rm -f /etc/apt/sources.list.d/cuda.list
RUN rm -f /etc/apt/sources.list.d/nvidia-ml.list

COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY src ${SRC_PATH}

ENV PYTHONPATH ${PYTHONPATH}:${SRC_PATH}
WORKDIR ${APP_PATH}

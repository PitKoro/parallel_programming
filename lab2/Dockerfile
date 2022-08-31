FROM nvidia/cuda:11.3.0-devel-ubuntu20.04

WORKDIR /app

RUN apt-get update -y
RUN apt-get install vim -y

RUN apt-get install -y python
RUN apt-get -y install python3-pip
RUN pip install Pillow

COPY ./src/ src/


FROM tensorflow/tensorflow:2.3.0

RUN apt-get update -y
COPY run_mnist.py requirements.txt /project/
WORKDIR /project
RUN pip install --no-cache-dir -r requirements.txt 

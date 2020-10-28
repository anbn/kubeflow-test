FROM tensorflow/tensorflow:2.3.0

RUN apt-get update -y
COPY requirements.txt /project/
WORKDIR /project
RUN pip install --no-cache-dir -r requirements.txt 
COPY run_mnist.py /project/

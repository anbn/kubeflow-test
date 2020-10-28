#!/bin/bash

docker build -t anbn-kube .
docker tag anbn-kube anbn1/anbn-kube
docker push anbn1/anbn-kube:latest

python3 pipeline.py test-pipeline.tgz

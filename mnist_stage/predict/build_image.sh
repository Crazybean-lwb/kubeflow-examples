#!/usr/bin/env bash

sudo docker build -f Dockerfile -t kubeflow/mnist-predict:v0.0.4 ./

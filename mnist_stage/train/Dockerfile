FROM harbor-qzm.stonewise.cn/proxy_cache/kubeflow/tensorflow:1.14.0-py3.6-cpu

COPY requirements.txt ./
ADD predict.py .
RUN pip3 install -r requirements.txt --index-url https://mirrors.aliyun.com/pypi/simple --extra-index-url https://wheels.galaxyproject.org/simple
ENTRYPOINT ["python3", "predict.py"]
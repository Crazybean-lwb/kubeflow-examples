FROM harbor-qzm.stonewise.cn/proxy_cache/kubeflow/tensorflow:1.14.0-py3.6-cpu
ADD load_data.py .
ENTRYPOINT ["python3", "load_data.py"]
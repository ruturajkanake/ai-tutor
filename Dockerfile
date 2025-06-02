FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/reqs.txt
RUN pip3 install --no-cache-dir -r /tmp/reqs.txt

COPY . /opt/rag-api
WORKDIR /opt/rag-api
EXPOSE 8081

CMD ["bash", "scripts/start.sh", "8081"]

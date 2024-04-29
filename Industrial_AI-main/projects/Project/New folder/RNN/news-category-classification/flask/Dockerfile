FROM ubuntu:20.04

WORKDIR /src/app

ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# apt update
RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata g++ curl

# install java
RUN apt-get install -y openjdk-8-jdk
ENV JAVA_HOME="/usr/lib/jvm/java-1.8-openjdk"

# install python and pip
RUN apt-get install -y python3-pip python3-dev
RUN cd /usr/local/bin && \
    ln -s /usr/bin/python3 python && \
    ln -s /usr/bin/pip3 pip && \
    pip install --upgrade pip

# apt clean
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install flask
RUN pip install pandas
RUN pip install tensorflow
RUN pip install konlpy

COPY . .

EXPOSE 15000

CMD ["python", "app.py"]
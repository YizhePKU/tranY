FROM ubuntu:21.04

WORKDIR /workspace

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

RUN apt-get install -y openjdk-11-jdk curl rlwrap
RUN curl -O https://download.clojure.org/install/linux-install-1.10.3.967.sh
RUN chmod +x linux-install-1.10.3.967.sh
RUN ./linux-install-1.10.3.967.sh

RUN apt-get install -y python3.9 python3-pip
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Download PyTorch first to speed up image rebuild.
RUN pip install torch==1.9.0

COPY deps.edn .
RUN clj -P

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["clojure", "-M", "-m", "nrepl.cmdline", "--middleware", "[cider.nrepl/cider-middleware]", "--bind", "0.0.0.0", "--port", "8848"]
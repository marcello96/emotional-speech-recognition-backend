FROM python:3.6-slim

MAINTAINER Marcin Jakubowski

RUN mkdir -p /project
WORKDIR /project

RUN apt-get update && apt-get install -y \
    nginx \
    supervisor \
    gcc \
    musl-dev \
 && rm -rf /var/lib/apt/lists/*

ADD /conf/supervisor.ini /etc/supervisor/supervisord.conf
ADD /conf/nginx.conf /etc/nginx/nginx.conf

RUN pip install --no-cache-dir cython
RUN pip install --no-binary :all: falcon

COPY requirements.txt /project/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /project

EXPOSE 80
CMD ["supervisord", "-n"]
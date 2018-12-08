FROM python:3.6

MAINTAINER Marcin Jakubowski

RUN apt-get update && \
    apt-get install -y nginx supervisor

ADD /conf/supervisor.ini /etc/supervisor/supervisord.conf
ADD /conf/nginx.conf /etc/nginx/nginx.conf

RUN mkdir -p /project
WORKDIR /project

RUN pip install cython
RUN pip install --no-binary :all: falcon

COPY requirements.txt /project/

RUN pip install --no-cache-dir -r requirements.txt

COPY ./src /project

EXPOSE 80
CMD ["supervisord", "-n"]
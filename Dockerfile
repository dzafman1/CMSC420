#Ubuntu base image with python 2.7
FROM ubuntu:18.04
COPY . /app
WORKDIR /app

#Install pip for Python 2.x
RUN apt-get update && apt-get install -y python-pip
RUN apt-get update && apt-get install -y curl
RUN curl -sL https://deb.nodesource.com/setup_8.x | bash -
RUN apt-get update && apt-get install -y nodejs

#----Intalling Polymer CLI----
RUN npm install npm@latest -g
RUN npm install -g bower
RUN npm install polymer-cli

#Installing python dependencies 
#RUN pip install -r ./requirements.txt

#downloads necessary packages
#RUN bower install --allow-root
EXPOSE 5000
#CMD python ./server/src/run.py

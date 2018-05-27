FROM ubuntu:latest

RUN apt-get update -y && apt-get install -y python3 python3-pip # pip3 install numpy jupyter scipy sklearn pandas tensorflow

RUN pip3 install jupyter

RUN useradd -ms /bin/bash jupyter

USER jupyter

WORKDIR /home/jupyter

CMD ["jupyter", "notebook", "--ip=*", "--no-browser"]


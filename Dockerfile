FROM tensorflow/tensorflow:latest-gpu

# update
RUN apt-get update && apt-get install -y \
    zip \ 
    unzip

# Upgrade PIP
RUN pip3 install --upgrade pip
# Install python packages
COPY requirements.txt requirements.txt
RUN pip3.8 install -r ./requirements.txt

# Clone rch-seg repo 
WORKDIR $HOME
RUN git clone git@github.com:JakobDomislovic/rch-seg.git

RUN cd rch-seg
RUN mkdir models
RUN mkdir data

WORKDIR $HOME/rch_seg
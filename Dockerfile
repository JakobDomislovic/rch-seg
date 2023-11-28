FROM tensorflow/tensorflow:latest-gpu

# update
RUN apt-get update && apt-get install -y \
    zip \ 
    unzip \
    git \
    nano

# Upgrade PIP
RUN pip3 install --upgrade pip
# Install python packages
WORKDIR $HOME
RUN pip install \
    numpy==1.26.2 \
    scipy==1.11.4 \
    gdown==4.7.1 \
    zipfile36==0.1.3 \
    albumentations==1.3.1 \
    matplotlib==3.8.2 \
    datetime==5.3 \
    tensorflow==2.15.0 \
    keras==2.15.0 \
    keras-applications==1.0.8 \
    segmentation-models==1.0.1 \
    h5py==3.10.0 \
    wandb==0.16.0 \
    pandas==2.1.3 \
    tf-clahe==0.1.0

# Clone rch-seg repo 
WORKDIR $HOME
RUN git clone https://github.com/JakobDomislovic/rch-seg.git

WORKDIR $HOME/rch-seg
RUN mkdir models
RUN mkdir data

WORKDIR $HOME/rch-seg
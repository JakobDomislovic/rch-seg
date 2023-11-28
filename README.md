# RCH-SEG

Clone repository from git:
```sh
git clone git@github.com:JakobDomislovic/rch-seg.git
```
If you want to make changes I suggest to create your own branch. If you want to merge please create pull request and add me as reviewer.

## Docker

Move to cloned repository:
```sh
cd rch-seg
```
Build docker with:
```sh
./docker_build.sh
```

Next step is to run docker image. When running docker image the cloned repository is mapped to docker container so you can change code locally and it will translate to docker container.

```sh
./docker_run.sh
```

## Train model

First step should be to put data into ```./data/``` direcotry. This will be mounted to docker container and wont be pushed to github.

Before neural network training, you should set up config ```config/train_config.yaml```  in which you can choose which data to train, view (both, view A or view b) will you use wandb (weights & biases), images per patients (600 if view==both and 300 if view=view A or view B), loss function and many more. For any question please contact me on jakob.domislovic@fer.hr. 

Training is started with ```python3 train.py```.

### TODO
- Upute za spajanje organa.
- Dodati support za smanjivanje slike na 96x96.
- Dodati support za augmentacije.
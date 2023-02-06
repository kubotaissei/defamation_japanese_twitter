# Defamation Detection Japanese
# 日本語誹謗中傷検出

- Build Envirionment

```bash
docker build . -f docker/Dockerfile -t defamation:cuda11
```

- Download and Preprocess Data 
```bash
sh bin/download_and_preprocess.sh TWITTER_BEARER_TOKEN
```

- Train

```bash
docker run --rm -it \
    --gpus '"device=0"' \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.ssh/:/root/.ssh \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --ipc=host \
    defamation:cuda11 \
    bash

python src/main.py data.type=target data.label_type=soft data.agg_type=gl model.n_msd=5 loss.params.gamma=0 model.num_classes=3
python src/main.py data.type=label data.label_type=soft data.agg_type=ds model.n_msd=6 loss.params.gamma=0 model.num_classes=4
```


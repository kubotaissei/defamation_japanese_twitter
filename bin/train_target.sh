docker run --rm -it \
    --gpus '"device=0"' \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.ssh/:/root/.ssh \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --ipc=host \
    defamation:cuda11 \
    python src/main.py data.type=target data.label_type=soft data.agg_type=gl model.n_msd=5 loss.params.gamma=0 model.num_classes=3
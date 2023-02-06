docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.ssh/:/root/.ssh \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --ipc=host \
    defamation:cuda11 \
    python src/main.py data.type=label data.label_type=soft data.agg_type=ds model.n_msd=6 loss.params.gamma=10 model.num_classes=4
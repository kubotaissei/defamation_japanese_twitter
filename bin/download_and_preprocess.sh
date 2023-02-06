docker run --rm -it \
    -v $PWD/:/root/workdir/ \
    -v $HOME/.ssh/:/root/.ssh \
    -v $HOME/.config/:/root/.config \
    -v $HOME/.netrc/:/root/.netrc \
    -v $HOME/.cache/:/root/.cache \
    --ipc=host \
    defamation:cuda11 \
    cd root/workdir & python scripts/download_and_preprocess.py $0
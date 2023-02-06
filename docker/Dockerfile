FROM  nvidia/cuda:11.0-base

ARG PYTHON_VERSION=3.9

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub

ENV TZ Asia/Tokyo
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja_JP:ja
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update \
     && apt-get install --yes --no-install-recommends \
     curl \
     git \
     wget \
     && apt-get clean

ENV PATH=/opt/conda/bin:$PATH

RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p /opt/conda \
 && rm ~/miniconda.sh \
 && conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include cython typing \
 && conda clean -ya

RUN pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

RUN  pip install opencv-python \
     tqdm \
     joblib \
     pytorch-lightning \
     hydra-core \
     hydra_colorlog \
     seaborn \
     plotly \
     h5py \
     wandb \
     jupyterlab \
     jupyterlab_code_formatter \
     black \
     mypy \
     lightgbm \
     flake8 \
     timm \
     xgboost \
     pydicom \
     rising \
     threadpoolctl \
     scikit-learn \
     kaggle \
     transformers==4.26 \
     datasets==2.8.0 \
     tokenizers \
     fugashi \
     ipadic \
     sentencepiece \
     crowd-kit

WORKDIR /root/workdir
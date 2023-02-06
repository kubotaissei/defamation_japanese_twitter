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
sh bin/train_target.sh
```

```bash
sh bin/train_label.sh
```


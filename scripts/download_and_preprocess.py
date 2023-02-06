import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import requests
from crowdkit.aggregation import GLAD, DawidSkene, MajorityVote
from datasets import load_dataset


def create_url(ids: list):
    tweet_fields = "tweet.fields=created_at"
    ids = f"ids={','.join(ids)}"
    url = "https://api.twitter.com/2/tweets?{}&{}".format(ids, tweet_fields)
    return url


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """

    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2TweetLookupPython"
    return r


def connect_to_endpoint(url):
    response = requests.request("GET", url, auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception(
            "Request returned an error: {} {}".format(
                response.status_code, response.text
            )
        )
    return response.json()


def get_text_data(examples):
    url = create_url(examples["id"])
    json_response = connect_to_endpoint(url)
    # print(json_response["data"])
    text_dict = {data["id"]: data["text"] for data in json_response["data"]}
    time_dict = {data["id"]: data["created_at"] for data in json_response["data"]}
    return {
        "text": [text_dict.get(id) for id in examples["id"]],
        "created_at": [time_dict.get(id) for id in examples["id"]],
    }


def get_dataset(train_path="input/train.pkl", test_path="input/test.pkl"):
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_pickle(train_path), pd.read_pickle(test_path)
    else:
        dataset = load_dataset("kubota/defamation-japanese-twitter")
        dataset = dataset.map(get_text_data, batched=True, batch_size=100)

        # 欠損(元ツイートが削除されているもの)を削除
        df = dataset["train"].to_pandas().dropna()
        # 全員がことなるもの，2名以上がCを選択したものを排除
        df = df[
            df["label"].apply(
                lambda l: np.median(l) != 0.0 if len(set(l)) != len(l) else False
            )
        ]
        # ラベル統合のために変形
        d_target = dict(
            worker=pd.concat(
                [df["user_id_list"].apply(lambda x: x[i]) for i in range(3)]
            ),
            task=df["id"].to_list() * 3,
            label=pd.concat([df["target"].apply(lambda x: x[i]) for i in range(3)]),
        )
        d_target = pd.DataFrame.from_dict(d_target)
        d_target["label"].replace({3: 0}, inplace=True)
        d_label = dict(
            worker=pd.concat(
                [df["user_id_list"].apply(lambda x: x[i]) for i in range(3)]
            ),
            task=df["id"].to_list() * 3,
            label=pd.concat([df["label"].apply(lambda x: x[i]) for i in range(3)]),
        )
        d_label = pd.DataFrame.from_dict(d_label)
        d_label["label"].replace({4: 0}, inplace=True)

        # 誹謗中傷の対象，種類ごとにそれぞれラベル統合
        df["mv_hard_target"] = list(MajorityVote().fit_predict(d_target))
        df["ds_hard_target"] = list(DawidSkene(n_iter=200).fit_predict(d_target))
        df["gl_hard_target"] = list(GLAD(n_iter=200).fit_predict(d_target))
        df["mv_soft_target"] = (
            MajorityVote().fit_predict_proba(d_target).to_numpy().tolist()
        )
        df["ds_soft_target"] = (
            DawidSkene(n_iter=200).fit_predict_proba(d_target).to_numpy().tolist()
        )
        df["gl_soft_target"] = (
            GLAD(n_iter=200).fit_predict_proba(d_target).to_numpy().tolist()
        )
        df["mv_hard_label"] = list(MajorityVote().fit_predict(d_label))
        df["ds_hard_label"] = list(DawidSkene(n_iter=200).fit_predict(d_label))
        df["gl_hard_label"] = list(GLAD(n_iter=200).fit_predict(d_label))
        df["mv_soft_label"] = (
            MajorityVote().fit_predict_proba(d_label).to_numpy().tolist()
        )
        df["ds_soft_label"] = (
            DawidSkene(n_iter=200).fit_predict_proba(d_label).to_numpy().tolist()
        )
        df["gl_soft_label"] = (
            GLAD(n_iter=200).fit_predict_proba(d_label).to_numpy().tolist()
        )

        # 学習データ，テストデータにそれぞれ分割
        train_df = df.query("created_at < '2022-05-21 00:00:00+00:00'").reset_index(
            drop=True
        )
        test_df = df.query("created_at > '2022-05-21 00:00:00+00:00'").reset_index(
            drop=True
        )
        train_df.to_pickle(train_path)
        test_df.to_pickle(test_path)
        return train_df, test_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("token", help="Twitter BEARER_TOKEN")
    args = parser.parse_args()
    BEARER_TOKEN = args.token
    train_df, test_df = get_dataset()

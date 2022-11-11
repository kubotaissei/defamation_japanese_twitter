from factory import read_yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def main():
    destination = "../input/data.pkl"
    df = pd.read_pickle(destination)
    df["resA_label"] = (
        df[list(filter(lambda x: "resA" in x, df.columns))]
        .apply(lambda d: sum(list(d), []), axis=1)
        .apply(lambda l: max(set(l), key=l.count) if len(set(l)) != len(l) else "None")
    )
    df["resB_label"] = (
        df[list(filter(lambda x: "resB" in x, df.columns))]
        .apply(lambda d: sum(list(d), []), axis=1)
        .apply(lambda l: max(set(l), key=l.count) if len(set(l)) != len(l) else "None")
    )
    # display(df.head())

    TEXT_COLUMN = "textDisplay"
    LABEL_COLUMN = "resB_label"

    data = df.query("resA_label != 'None' & resB_label != 'None'")[
        [TEXT_COLUMN, LABEL_COLUMN]
    ].reset_index(drop=True)
    print(data.head())
    print(data.shape)
    print(data.groupby(LABEL_COLUMN).count())

    le = LabelEncoder()
    data[LABEL_COLUMN] = le.fit_transform(data[LABEL_COLUMN])
    train_df, test_df = train_test_split(
        data, test_size=0.1, random_state=1, stratify=data[LABEL_COLUMN]
    )
    train_df.reset_index(drop=True).to_pickle("../input/train.pkl")
    test_df.reset_index(drop=True).to_pickle("../input/test.pkl")


if __name__ == "__main__":
    main()

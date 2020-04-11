import keras
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import numpy as np

def preprocess(data,max_len,**args):
    df = data.fillna(method='ffill')


    ##### Get sentences
    agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                  s['POS'].values.tolist(),
                                                  s['Tag'].values.tolist())]
    grouped = df.groupby("Sentence #").apply(agg)
    sentences = [s for s in grouped]

    # Getting unique words and labels from data
    words = list(df['Word'].unique())
    tags = list(df['Tag'].unique())
    # Dictionary word:index pair
    # word is key and its value is corresponding index
    word_to_index = {w: i + 2 for i, w in enumerate(words)}
    word_to_index["UNK"] = 1
    word_to_index["PAD"] = 0

    # Dictionary lable:index pair
    # label is key and value is index.
    tag_to_index = {t: i + 1 for i, t in enumerate(tags)}
    tag_to_index["PAD"] = 0

    idx2word = {i: w for w, i in word_to_index.items()}
    idx2tag = {i: w for w, i in tag_to_index.items()}


    # Converting each sentence into list of index from list of tokens
    X = [[word_to_index[w[0]] for w in s] for s in sentences]

    # Padding each sequence to have same length  of each word
    X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=word_to_index["PAD"])

    # Convert label to index
    y = [[tag_to_index[w[2]] for w in s] for s in sentences]

    # padding
    y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag_to_index["PAD"])
    num_tag = df['Tag'].nunique()
    # One hot encoded labels
    y = np.array([to_categorical(i, num_classes=num_tag + 1) for i in y])
    return X,y,len(df['Word'].unique())+2

def test_pandas_fillna(data, **args):
    return data.fillna(**args)


def test_onehot_sentences(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, word_dict, sentence_groups: np.array(
                keras.preprocessing.sequence.pad_sequences(
                    [
                        [word_dict[x] for x in sw]
                        for sw in [y.values for _, y in sentence_groups["Word"]]
                    ],
                    ml,
                    padding="post",
                    value=0,
                    dtype="int",
                ),
                dtype="O",
            )
        )(
            data,
            max_len,
            {y: x for x, y in enumerate(["PAD", "UNK"] + list(data["Word"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


def test_word_count(data):
    return data["Word"].nunique() + 2


def test_word_categorical_labels_per_sentence(data, max_len):
    return (
        lambda df, max_len: (
            lambda d, ml, c, tag_dict, sentence_groups: np.array(
                [
                    keras.utils.to_categorical(i, num_classes=c + 1)
                    for i in keras.preprocessing.sequence.pad_sequences(
                        [
                            [tag_dict[w] for w in s]
                            for s in [y.values for _, y in sentence_groups["Tag"]]
                        ],
                        ml,
                        padding="post",
                        value=0,
                    )
                ]
            )
        )(
            data,
            max_len,
            data["Tag"].nunique(),
            {y: x for x, y in enumerate(["PAD"] + list(data["Tag"].unique()))},
            data.groupby("Sentence #"),
        )
    )(data, max_len)


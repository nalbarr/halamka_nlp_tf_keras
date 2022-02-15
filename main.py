import pandas as pd
import numpy as np
import random
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def dump_tf_info():
    print("Version: ", tf.__version__)
    print("Eager mode: ", tf.executing_eagerly())
    print(
        "GPU is",
        "available"
        if tf.config.experimental.list_physical_devices("GPU")
        else "NOT AVAILABLE",
    )


def load_data(file_path):
    df = pd.read_csv(file_path, sep="\t")
    print(df.head())
    return df


def remove_header(df):
    df = df[1:]
    return df


def remove_category_2_rows(df):
    df = df[(df.category == 0) | (df.category == 1)]
    return df


def get_nrows(df):
    (nrows, ncols) = df.shape
    return nrows


def get_nlp_hyperparameters(nrows):
    embedding_dim = 100
    max_length = 16
    trunc_type = "post"
    padding_type = "post"
    oov_tok = "<OOV>"
    training_size = int(0.9 * nrows)
    test_portion = 0.1
    return (
        embedding_dim,
        max_length,
        trunc_type,
        padding_type,
        oov_tok,
        training_size,
        test_portion,
    )


def get_corpus(df):
    corpus = []
    num_sentences = 0
    for index, row in df.iterrows():
        list_item = []
        list_item.append(row["title"])
        this_label = row["category"]

        if this_label == 0:
            list_item.append(0)
        elif this_label == 1:
            list_item.append(1)
        else:
            print("Unknown category.")

        num_sentences += 1
        corpus.append(list_item)

    print("num_sentences: {0}".format(num_sentences))
    print("len(corpus): {0}".format(len(corpus)))
    print("corpus[0]: {0}".format(corpus[0]))

    return num_sentences, corpus


def tokenize(corpus, test_portion, training_size, max_length, padding_type, trunc_type):
    sentences = []
    labels = []
    random.shuffle(corpus)
    for x in range(training_size):
        sentences.append(corpus[x][0])
        labels.append(corpus[x][1])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(sentences)
    word_index = tokenizer.word_index
    vocab_size = len(word_index)

    sequences = tokenizer.texts_to_sequences(sentences)
    padded = pad_sequences(
        sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
    )

    split = int(test_portion * training_size)

    test_sequences = padded[0:split]
    training_sequences = padded[split:training_size]
    test_labels = labels[0:split]
    training_labels = labels[split:training_size]

    return (
        word_index,
        vocab_size,
        training_sequences,
        training_labels,
        test_sequences,
        test_labels,
    )


def get_embeddings_matrix(word_index, vocab_size, embedding_dim):
    embeddings_index = {}

    with open("/tmp/glove.6B.100d.txt") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = coefs

    embeddings_matrix = np.zeros((vocab_size + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embeddings_matrix[i] = embedding_vector

    return embeddings_matrix


def create_model(vocab_size, embedding_dim, max_length, embeddings_matrix):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size + 1,
                embedding_dim,
                input_length=max_length,
                weights=[embeddings_matrix],
                trainable=False,
            ),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Conv1D(64, 5, activation="relu"),
            tf.keras.layers.MaxPooling1D(pool_size=4),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())

    return model


def dump_input_types(training_sequences, training_labels, test_sequences, test_labels):
    print(
        "training_sequences: ",
        training_sequences.shape,
        type(training_sequences),
        training_sequences.dtype,
    )
    print("training_labels: ", type(training_labels))
    print(
        "test_sequences: ",
        test_sequences.shape,
        type(test_sequences),
        test_sequences.dtype,
    )
    print("test_labels: ", type(test_labels))


def convert_input_type(
    training_sequences, training_labels, testing_sequences, test_labels
):
    training_labels = np.array(training_labels)
    test_labels = np.array(test_labels)

    return training_sequences, training_labels, testing_sequences, test_labels


def train_model(
    model, training_sequences, training_labels, test_sequences, test_labels, num_epochs
):
    history = model.fit(
        training_sequences,
        training_labels,
        epochs=num_epochs,
        validation_data=(test_sequences, test_labels),
        verbose=2,
    )
    print(history)


def save_model(model):
    model.save("models/halamka_nlp_tf.h5")


def main():
    dump_tf_info()

    df = load_data("data/halamka_posts_1836.tsv")
    df = remove_header(df)
    df = remove_category_2_rows(df)
    nrows = get_nrows(df)

    (
        embedding_dim,
        max_length,
        trunc_type,
        padding_type,
        oov_tok,
        training_size,
        test_portion,
    ) = get_nlp_hyperparameters(nrows)
    num_sentences, corpus = get_corpus(df)
    (
        word_index,
        vocab_size,
        training_sequences,
        training_labels,
        test_sequences,
        test_labels,
    ) = tokenize(
        corpus, test_portion, training_size, max_length, padding_type, trunc_type
    )
    embeddings_matrix = get_embeddings_matrix(word_index, vocab_size, embedding_dim)

    model = create_model(vocab_size, embedding_dim, max_length, embeddings_matrix)
    dump_input_types(training_sequences, training_labels, test_sequences, test_labels)
    (
        training_sequences2,
        training_labels2,
        test_sequences2,
        test_labels2,
    ) = convert_input_type(
        training_sequences, training_labels, test_sequences, test_labels
    )

    train_model(
        model,
        training_sequences2,
        training_labels2,
        test_sequences2,
        test_labels2,
        num_epochs=50,
    )

    save_model(model)


if __name__ == "__main__":
    import time

    start_time = time.time()
    main()
    print("--- {} seconds ---".format(time.time() - start_time))

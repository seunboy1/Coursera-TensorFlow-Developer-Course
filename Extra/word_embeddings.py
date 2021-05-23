import os
import re
import shutil
import string
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten, Dropout


def download_data():

    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

    dataset = tf.keras.utils.get_file("aclImdb_v1.tar.gz", url,
                                      untar=True, cache_dir='.',
                                      cache_subdir='')

    data_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

    return data_dir


def create_data(dir):
    sentences = []
    labels = []
    x = ["neg", "pos"]
    for label in x:
        path = os.path.join(dir, label)
        txts = os.listdir(path)
        for txt in txts:
            data = open(os.path.join(path, txt)).read()
            sentences.append(data)
            labels.append(np.array(x.index(label)))

    return sentences, np.array(labels)

def preprocess(input_data):
    lowercase = [x.lower() for x in input_data]
    stripped_html = [re.sub("<br />", " ", x) for x in lowercase]
    data = [s.translate(str.maketrans('', '', string.punctuation)) for s in stripped_html]
    return data


if __name__ == '__main__':

    # data_dir = download_data()
    dataset_dir = "../Tensorflow_Exam/aclImdb"
    train_dir = "../Tensorflow_Exam/aclImdb/train"
    test_dir = "../Tensorflow_Exam/aclImdb/test"

    vocab_size = 10000
    embedding_dim = 32
    max_length = 200
    batch_size = 32
    num_epochs = 10
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    validation_sentences, validation_labels = create_data(test_dir)
    train_sentences, train_labels = create_data(train_dir)

    train_sentences = preprocess(train_sentences)
    validation_sentences = preprocess(validation_sentences)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(train_sentences)
    word_index = tokenizer.word_index

    train_sequences = tokenizer.texts_to_sequences(train_sentences)
    train_padded = pad_sequences(train_sequences, padding=padding_type, maxlen=max_length)

    val_sequences = tokenizer.texts_to_sequences(validation_sentences)
    val_padded = pad_sequences(val_sequences, maxlen=max_length)

    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        Dropout(0.5),
        GlobalAveragePooling1D(),  # Flatten(), #
        Dropout(0.8),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    callback = tf.keras.callbacks.TensorBoard(log_dir="word_embeddingd_logs")
    history = model.fit(train_padded, train_labels, epochs=num_epochs, validation_data=(val_padded, validation_labels),
                        callbacks=[callback])

    model.save('word_embeddings.h5')
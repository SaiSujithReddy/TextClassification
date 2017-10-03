from data_processing import *

print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in word2vec.vocab:
        embedding_matrix[i] = word2vec.word_vec(word)
    else:
        embedding_matrix[i] = np.random.rand(300)
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))


#LSTM cells should be proportionate to dataset size: 200 cells ~100k dataset
model = Sequential()
model.add(Embedding(nb_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=False))
model.add(Bidirectional(LSTM(EMBEDDING_DIM, dropout_W=0.2, recurrent_dropout=0.4)))
model.add(Dense(10, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Done with building model")

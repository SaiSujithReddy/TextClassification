from data_processing import *

print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

#LSTM cells should be proportionate to dataset size: 200 cells ~100k dataset
model = Sequential()
model.add(Embedding(nb_words, 128))
model.add(LSTM(EMBEDDING_DIM, dropout_W=0.2, recurrent_dropout=0.4))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(10, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("Done with building model")

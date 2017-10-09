from data_processing import *

def model_config():
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
	return model

def model_fit(model,train_data_1,Y_train,test_data_1,Y_test,class_weight):
	print('Evaluating the model ...')
	model.fit(train_data_1, Y_train,
          batch_size=64,
          epochs=1,
          validation_data=(test_data_1, Y_test),class_weight=class_weight)
score, acc = model.evaluate(test_data_1, Y_test,
                            batch_size=64)
	print('Test score:', score)
	print('Test accuracy:', acc)
	model.save("model.h5")

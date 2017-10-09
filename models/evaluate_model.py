from build_model import *

print('Evaluating the model ...')
model.fit(train_data_1, Y_train,
          batch_size=64,
          epochs=1,
          validation_data=(test_data_1, Y_test),class_weight=class_weight)
score, acc = model.evaluate(test_data_1, Y_test,
                            batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)


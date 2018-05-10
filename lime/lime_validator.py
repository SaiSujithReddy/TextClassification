from lime.lime_text import LimeTextExplainer
from utils.helper_functions import *
from utils.constants import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from data.data_processing import *


labels = [x.replace('skill_name_', '') for x in list(Y_train.columns)]
explainer = LimeTextExplainer(class_names=labels)
idx = 1204
X_test = X_test.reset_index(drop=True)
model=load_model("../models/model.h5")

def classifier(X):
    X_1 = [text_to_wordlist(x) for x in X]
    test_sequences_1 = tokenizer.texts_to_sequences(X_1)
    test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
    return model.predict(test_data_1)

def get_explanation(X_test,validate):
    exp = explainer.explain_instance(X_test[idx], classifier, num_features=6, labels=[0, 9])
    print('Document id: %d' % idx)
    print('Predicted class =', validate['skill_name_pred'][idx])
    print('True class: %s' % validate['skill_name_test'][idx])
    print ('Explanation for class %s' % labels[0])
    print ('\n'.join(map(str, exp.as_list(label=0))))

    print ('Explanation for class %s' % labels[9])
    print ('\n'.join(map(str, exp.as_list(label=9))))

    exp = explainer.explain_instance(list(X_test)[idx], classifier, num_features=6, top_labels=1)
    print(exp.available_labels())
    exp.show_in_notebook(text=X_test[idx])


def get_examples(predictions):
    good_list = []
    for i in range(0, 10000):
        if predictions[i].max() > 0.4:
            good_list.append(i)

    for i in good_list[0:15]:
        exp = explainer.explain_instance(list(X_test)[i], classifier, num_features=6, top_labels=1)
        exp.show_in_notebook(text=X_test[i])

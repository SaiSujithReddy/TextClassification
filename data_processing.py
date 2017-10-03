from helper_functions import *

VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 200000
EMBEDDING_DIM = 200

df = read_data()
df = clean_up(df)
df = drop_head(drop_tail(df))
df = df[~(df['skill_name']=='Attempts to upsell')]
df.replace(to_replace={'skill_name':{'Probing question':'Fact-gathering question'}},inplace=True)

X_train, X_test, Y_train, Y_test= create_train_test(df,VALIDATION_SPLIT)
Y_train_unique = np.unique(Y_train)
class_weight = class_weight.compute_class_weight('balanced', Y_train_unique, Y_train)

Y_train=get_dummies(Y_train)
Y_test=get_dummies(Y_test)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(list(df['content']))

word_index = tokenizer.word_index

train_sequences_1 = tokenizer.texts_to_sequences(list(X_train))
train_data_1 = pad_sequences(train_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

test_sequences_1 = tokenizer.texts_to_sequences(list(X_test))
test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)

Y_train=np.matrix(Y_train)
Y_test=np.matrix(Y_test)

print("Done with data processing")

import csv
from src.utils import *
import torch as th
import pandas as pd
import numpy as np
from src.models import Conv1dClassifier
from src.models import LSTMClassifier
import json
import zipfile

# Data exctracting
print("Data extracting...") 
with zipfile.ZipFile("data/train/train.csv.zip","r") as zip_ref:
	zip_ref.extractall("data/train/")
with zipfile.ZipFile("data/train/test.csv.zip","r") as zip_ref:
	zip_ref.extractall("data/train/")
with zipfile.ZipFile("data/train/test_labels.csv.zip","r") as zip_ref:
	zip_ref.extractall("data/train/")
with zipfile.ZipFile("data/train/glove.6B.zip","r") as zip_ref:
	zip_ref.extractall("data/train/")
print()

# Data loading
print("Data loading...")
S_train, y_train, labels = loadTexts("data/train/train.csv", None, limit=-1, maxLength=1000)
S_test, y_test, _ = loadTexts("data/train/test.csv", "data/train/test_labels.csv", limit=-1, maxLength=-1)
embd = pd.read_csv("data/train/glove.6B.100d.txt", sep=" ", error_bad_lines=False, index_col=0, engine="python", names=range(100), quoting=csv.QUOTE_NONE)
print()

# Data processing
print("Data processing...")
w2idx, X = text_to_index(S_train + S_test)
vocab = list(w2idx.keys())
X_train = X[:len(y_train)]
X_test = X[len(y_train):][:10]
y_test = y_test[:10]
weights = get_weights(vocab, embd)
print()


# Training
neutral_ids = np.where(np.sum(y_train,axis=1)==0)[0]
other_ids = np.where((y_train[:,0]==1) & (y_train[:,1]+y_train[:,3]+y_train[:,5]==0))[0]
ids = list(set(range(len(X_train))) - set(np.random.choice(neutral_ids, int(0.95*len(neutral_ids)), False)) - set(other_ids)) # balancing

ids = np.random.choice(ids, len(ids), False) # shuffling

print("CNN model training...")
cnn = Conv1dClassifier(vocab_size=len(vocab), n_labels=len(labels), embedding_dim=50, kernels=[3,4,5], maps=100, embedding_weights=th.Tensor(weights))
loss_fun = th.nn.BCELoss()
optimizer = th.optim.Adam(cnn.parameters(), 0.0002)
history = cnn.fit(X=[X_train[i] for i in ids], y=y_train[ids], optimizer=optimizer, loss_fun=loss_fun, batch_size=100, epochs=1)

print("LSTM model training...")
lstm = LSTMClassifier(vocab_size=len(vocab), n_labels=len(labels), embedding_dim=100, hidden_dim=50, feature_dim=40, embedding_weights=th.Tensor(weights))
loss_fun = th.nn.BCELoss()
optimizer = th.optim.Adam(lstm.parameters(), 0.0002)
history = lstm.fit(X=[X_train[i] for i in ids], y=y_train[ids], optimizer=optimizer, loss_fun=loss_fun, batch_size=100, epochs=1)
print()


# Test
print("CNN model testing...")
acc, per_label_acc = cnn.evaluate([X_test[i] for i in range(len(X_test)) if y_test[i,0]!=-1], y_test[y_test[:,0]!=-1], 200)
print("Accuracies: ")
print("	mean:", round(acc, 3))
for i in range(len(labels)):
	print("	",labels[i], ":", round(per_label_acc[i],3))
print()
print("LSTM model testing...")
acc, per_label_acc = lstm.evaluate([X_test[i] for i in range(len(X_test)) if y_test[i,0]!=-1], y_test[y_test[:,0]!=-1], 200)
print("Accuracies: ")
print("	mean:", round(acc, 3))
for i in range(len(labels)):
	print("	",labels[i], ":", round(per_label_acc[i],3))
print()


# Saving
print("Models saving...")
weights.tofile("data/test/weights.npy")

with open('data/test/w2idx.json', 'w') as fp:
    json.dump(w2idx, fp)

th.save(cnn.state_dict(), "data/test/cnn.pt")
th.save(lstm.state_dict(), "data/test/lstm.pt")

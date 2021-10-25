import torch as th
import numpy as np
import json
from src.models import Conv1dClassifier
from src.models import LSTMClassifier
from src.utils import get_indices
from src.utils import plot_cmap
from src.utils import clean_str

# Model loading
print("Data loading...")
with open('data/test/w2idx.json', 'r') as fp:
    w2idx = json.load(fp)
weights = np.fromfile("data/test/weights.npy").reshape((len(w2idx.keys()), -1))

labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
print()

print("Models loading...")
cnn = Conv1dClassifier(vocab_size=len(w2idx.keys()), n_labels=len(labels), embedding_dim=50, kernels=[3,4,5], maps=100, embedding_weights=th.Tensor(weights))
cnn.load_state_dict(th.load("data/test/cnn.pt"))
cnn.eval()

lstm = LSTMClassifier(vocab_size=len(w2idx.keys()), n_labels=len(labels), embedding_dim=100, hidden_dim=50, feature_dim=40, embedding_weights=th.Tensor(weights))
lstm.load_state_dict(th.load("data/test/lstm.pt"))
lstm.eval()

print()
# Test
f = open("data/test/test.txt", "r")
comments = f.readlines()
for i,comment in enumerate(comments):
	print("####")
	s = clean_str(comment).split()
	x = get_indices([s], w2idx)[0]
	print(comment)
	print("CNN prediction : ", cnn.predict(x.unsqueeze(0)))
	print("LSTM prediction : ", lstm.predict(x.unsqueeze(0)))
	plot_cmap(cnn, x, s, labels, "figure/comment"+str(i))
	print()
f.close()

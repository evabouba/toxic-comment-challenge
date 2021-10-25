from sklearn import metrics
import torch as th
import numpy as np
from .utils import get_batches


class TextClassifier(th.nn.Module):
	def __init__(self, vocab_size, embedding_dim=20, embedding_weights=None):
		super(TextClassifier, self).__init__()

		self.embd_list = th.nn.ModuleList()
		self.embd_list.append(th.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)) # padding_idx=0 : word 0 is embedded by a null vector (for batch training)
		self.embd_dim = embedding_dim
		if embedding_weights is not None:
			self.embd_dim += embedding_weights.shape[1]
			self.embd_list.append(th.nn.Embedding.from_pretrained(embedding_weights, freeze=True, padding_idx=0))

	def fit(self, X, y, optimizer, loss_fun, batch_size=1, epochs=1, X_val=None, y_val=None):
		history = {"loss":[], "mean_acc":[]}
		n_batch = np.ceil(len(X) / batch_size)
		n_val_batch = 0
		if X_val is not None:
			n_val_batch = np.ceil(len(X_val) / 200)
			history["val_loss"] = []
			history["val_mean_acc"] = []
 
		for e in range(1, epochs+1):
			# batches
			X_batch, y_batch = get_batches(X, y, batch_size, 2)
			X_val_batch, y_val_batch = None, None
			if X_val is not None:
				X_val_batch, y_val_batch = get_batches(X_val, y_val, 200, 2)

			# train
			loss = 0
			mean_acc = 0 
			for i in range(int(n_batch)):
				y_pred = self(X_batch[i])
		  
				l = (loss_fun(y_pred, y_batch[i])*(1+y_batch[i])).mean()
				optimizer.zero_grad()
				l.backward()
				optimizer.step()
 
				loss += l.item()
				mean_acc += (th.round(y_pred) == y_batch[i]).to(th.float32).mean().item()

			# validation
			val_loss = 0
			val_mean_acc = 0
			if X_val is not None:
				for i in range(int(n_val_batch)):
					y_pred = self(X_val_batch[i])
					val_loss += (loss_fun(y_pred, y_val_batch[i])*(1+y_val_batch[i])).mean().item()
					val_mean_acc += (th.round(y_pred) == y_val_batch[i]).to(th.float32).mean().item()

			# print
			log = "loss: {:.3f} - acc: {:.3f}".format(loss/n_batch, mean_acc/n_batch)
			history["loss"].append(loss/n_batch)
			history["mean_acc"].append(mean_acc/n_batch)
			if X_val is not None:
				log += " - val_loss: {:.3f} - val_acc: {:.3f}".format(val_loss/n_val_batch, val_mean_acc/n_val_batch)
				history["val_loss"].append(val_loss/n_val_batch)
				history["val_mean_acc"].append(val_mean_acc/n_val_batch)
			print(log)

		return history

	def predict(self, X, batch_size=1):
		# batch_size : number of samples predicted at once (too many can cause memory problems)
		# return a prediction tensor
		X_batch, _ = get_batches(X, None, batch_size, 0)
		pred = []

		self.eval() 
		for i in range(len(X_batch)):
			pred += self(X_batch[i]).tolist()
		self.train()

		return th.Tensor(pred) 
 
	def evaluate(self, X, y, batch_size=1):
		# batch_size : size of batch for prediction (too large can cause memory problems)
		# return the model mean label-wise accuracy on samples (X,y) 
		y_pred = self.predict(X, batch_size)
		per_label_acc = (th.round(y_pred) == th.Tensor(y)).to(th.float32).mean(axis=0)
		mean_acc = per_label_acc.mean().item()
  
		return mean_acc, per_label_acc.tolist()
 
	def metrics(self, X, y, batch_size):
		# return lists of increasing false positive rate and true positive rate for per label ROC curves

		y_pred = np.array(self.predict(X, batch_size))
		curves = {"fpr":[], "tpr":[], "precision":[], "recall":[]}

		for i in range(y.shape[1]):
			f, t, _ = metrics.roc_curve(y[:,i], y_pred[:,i], 1)
			p, r, _ = metrics.precision_recall_curve(y[:,i], y_pred[:,i], 1)
		
			curves["fpr"].append(f)
			curves["tpr"].append(t)
			curves["precision"].append(p)
			curves["recall"].append(r)

		return curves

  
class Conv1dClassifier(TextClassifier):
	'''A text classifier:
	- input : a list of word indices
	- output : probability associated to a binary classification task
	- vocab_size : the number of words in the vocabulary we want to embed
	- n_labels : number of labels to predict
	- embedding_dim : dimension of trainable word embeddings
	_ kernels : list of convolution kernel sizes
	_ maps : number of feature maps per kernel size
	- embedding_weights : not trainable pre-trained word embeddings of shape (vocab_size, k) to be concatenated with trainable embeddings
	'''
	def __init__(self, vocab_size, n_labels, embedding_dim=20, kernels=[3], maps=10, embedding_weights=None):
		super(Conv1dClassifier, self).__init__(vocab_size, embedding_dim, embedding_weights)
	  

		self.conv_list = th.nn.ModuleList()
		for k in kernels:
			self.conv_list.append(th.nn.Conv1d(in_channels=self.embd_dim, out_channels=maps, kernel_size=k, padding=int(np.floor((k-1)/2)), bias=False))
		
		self.dropout = th.nn.Dropout(0.3)
		self.linear = th.nn.Linear(len(kernels)*maps, n_labels)
		self.activation = th.nn.Sigmoid()
 
	def feature_map(self, X, hook=False):
		# hook : whether to register the hook or not
		embds = []
		for embd in self.embd_list:
			embds.append(embd(X).permute(0,2,1))
		embds = th.cat(embds, dim=1) # concatenation of different embeddings

		maps = []
		for conv in self.conv_list:
			if conv.kernel_size[0]%2 == 0:
				map = conv(th.nn.functional.pad(embds, (0,1), value=0))
			else:
				map = conv(embds)
  
			if hook:
				h = map.register_hook(self.activations_hook)

			maps.append(map)
 
		return th.cat(maps, dim=1)

			
	def forward(self, input, hook=False):
		# hook : whether to register the hook or not

		map = th.relu(self.feature_map(input, hook))
	 
		feature_vec = th.max(map, dim=2)[0] # pooling over words
		dropout = self.dropout(feature_vec)
		dense = self.linear(dropout)
		output = self.activation(dense)
 
		return output

 
	def activations_hook(self, grad):
		if not hasattr(self, 'gradients'):
			self.gradients = []

		self.gradients.append(grad)
 
 
	def get_gradcam(self, x, c):
		# x : a sample
		# return the list of heatmaps of input for each label

		self.eval()
		pred = self(x.unsqueeze(0), hook=True)
		pred[:, c].backward()
 
		pooled_gradients = th.mean(th.cat(self.gradients, dim=1), dim=[0, 2]) # global average pooling of each filter
		activations = self.feature_map(x.unsqueeze(0)).detach() 

		for i in range(len(pooled_gradients)): # channels weighting
			activations[:, i, :] *= pooled_gradients[i]
 
		heatmap = th.sum(activations, dim=1) # pooling over filters
		heatmap = th.relu(heatmap)

		self.gradients = []
		self.train()
 
		return heatmap

class LSTMClassifier(TextClassifier):
	'''A text classifier:
	- input : a list of word indices
	- output : probability associated to a binary classification task
	- vocab_size : the number of words in the vocabulary we want to embed
	- n_labels : number of labels to predict
	- embedding_dim : dimension of trainable word embeddings
	- hidden_dim : dimension of the hidden state in the LSTM cell
	'''
	def __init__(self, vocab_size, n_labels, embedding_dim=20, hidden_dim=10, feature_dim=10, embedding_weights=None):
		super(LSTMClassifier, self).__init__(vocab_size, embedding_dim, embedding_weights)
		
		self.lstm = th.nn.LSTM(self.embd_dim, hidden_dim, batch_first=True)
		self.dropout = th.nn.Dropout(0.2)
		self.fc1 = th.nn.Linear(hidden_dim, feature_dim)
		self.act1 = th.nn.ReLU()
		self.fc2 = th.nn.Linear(feature_dim, n_labels)
		self.act2 = th.nn.Sigmoid()

	def forward(self, input):
		embds = []
		for embd in self.embd_list:
			embds.append(embd(input))
		embds = th.cat(embds, dim=2) # concatenation of different embeddings

		out, (hidden, cell) = self.lstm(embds)
		output = self.dropout(hidden[-1])
		output = self.fc1(output)
		output = self.act1(output)
		output = self.dropout(output)
		output = self.fc2(output)
		output = self.act2(output)

		return output

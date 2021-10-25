import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import torch as th


def clean_str(string, tolower=True):
    """
    Tokenization/string cleaning.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    if tolower:
        string = string.lower()
    return string.strip()
 
 
def loadTexts(filename, labels_filename=None, limit=-1, maxLength=-1):
    """
    Texts loader.
    If limit is set to -1, the whole dataset is loaded, otherwise limit is the number of lines
    If maxLenght is not -1, removal of the sentences with more than maxLength words
    """
    df = pd.read_csv(filename, index_col=0)
    S = []
    if labels_filename is None:
        y = []
    else:
        y = pd.read_csv(labels_filename, index_col=0).to_numpy()
    
    cpt=0
    skip=0
    for i in range(df.shape[0]) :
        cleanline = clean_str(df.iloc[i,0]).split()
        if cleanline and ((maxLength == -1) or (len(cleanline) < maxLength)): 
            S.append(cleanline)
            if labels_filename is None:
                y.append(list(df.iloc[i,1:]))

        else: 
            if labels_filename is not None:
                y = np.delete(y, cpt, axis=0)
            skip+=1
            continue
        cpt+=1 
        if limit > 0 and cpt >= limit: 
            break
               
    print("Load ", cpt, " lines from ", filename , " / ", skip ," lines discarded")
 
    return S, np.array(y), list(df.columns[1:])
 
def text_to_index(S): # create an index from sentences in S
    w2idx = {"PADDING":0} # word 0 is used to padd short sentences in batches
    X = []
    length = 1

    for sent in S: 
        isent = []
        for w in sent: 
            if not w in w2idx.keys():
                w2idx[w] = length
                isent.append(length)
                length += 1
            else:
                isent.append(w2idx[w])

        X.append(th.LongTensor(list(isent)))

    return w2idx, X

def get_indices(S, w2idx): # convert sentences into index based on the vocabulary, unknown words are treated as PADDING
    X = []

    for sent in S: 
        isent = []
        for w in sent: 
            if not w in w2idx.keys():
                isent.append(0)
            else:
                isent.append(w2idx[w])

        X.append(th.LongTensor(list(isent)))

    return X

def find_matching_index(list1, list2):
    inverse_index = { element: index for index, element in enumerate(list1) }

    return [(inverse_index[element], index)
        for index, element in enumerate(list2) if element in inverse_index]
        
def get_weights(vocab, embd): # map vocabulary words to its embedding, unknown words are given null embedding
    matches = find_matching_index(vocab, list(embd.index))
    weights = np.zeros((len(vocab), embd.shape[1]))

    for match in matches:
        weights[match[0]] = list(embd.iloc[match[1]])
  
    return weights

def plot_cmap(classifier, x, s, labels, save=None):
    plt.figure(figsize=(2,30))
    heatmap = []
    pred = np.round(classifier.predict(x.unsqueeze(0))[0].tolist(),2)
 
    for i in range(len(labels)):
        heatmap.append(classifier.get_gradcam(x, i).squeeze().tolist())
 
    plt.matshow(heatmap)
    for (i, j), k in np.ndenumerate(heatmap):
        plt.text(j, i, round(k,3), va="center", ha="center")

    plt.yticks(range(len(labels)), [(labels[i], pred[i]) for i in range(len(labels))])
    plt.xticks(range(len(x)), s, rotation=60);

    if save is not None:
        plt.savefig(save, bbox_inches="tight")

def get_batches(X, y=None, batch_size=1, method=0): # create a list of tensor batches by padding the shortest sentences in each batch
    # return a list of tensors of shape (batch_size, max {len(s) for s in batch}) by padding shorter sentences with zeros

    if method == 1: # random batches
        ids = np.random.choice(len(X), len(X), False)
    elif method == 2 : # batches with sentences of similar length (to limit padding)
        lengths = [len(x) for x in X]
        rand = np.random.random(len(lengths))
        ids = np.lexsort((rand, lengths)) # sentence order by length with same-length sentences shuffled
    else:  # unshuffled batches
        ids = np.arange(len(X))
 
    X_batch, y_batch = [], []
 
    for i in range(0, len(X), batch_size):
        max_length = max([len(X[ids[j]]) for j in range(i, min(len(X),i+batch_size))])
        batch_x = []
        batch_y = []
        for j in range(batch_size):
            if i+j < len(X):
                batch_x.append(list(th.cat([X[ids[i+j]], th.zeros(max_length-len(X[ids[i+j]]))])))
                if y is not None:
                    batch_y.append(y[ids[i+j]])
            else:  # if the number of samples is not a multiple of batch_size
                break
 
        y_batch.append(th.Tensor(batch_y))
        X_batch.append(th.LongTensor(batch_x))
 
    return X_batch, y_batch

# toxic-comment-challenge

This academic project is based on kaggle [toxic comment classification challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge). A convolutional network and a LSTM network are cmmpared. The convolutional network also serves as an automatic censor using [GradCam](https://arxiv.org/abs/1610.02391). It was co-developed with [Emma Amblard](https://github.com/emmaamblard) and [Charlotte Durand](https://github.com/cdurand95). 

## Running the example scripts
* train.py : training and saving of the two classifiers
* test.py : loading of the two pre-trained classifiers and prediction of the comments in the file data/test/test.txt

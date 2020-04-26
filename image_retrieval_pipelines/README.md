There are couple of pre-requisites for running the comparison evaluation jupyter notebook:
a. encoder folder should be populated with infersent1.pkl encoder. Make sure to run the below command from parent folder.
  - curl -Lo encoder/infersent1.pkl https://dl.fbaipublicfiles.com/infersent/infersent1.pkl
b. GloVe folder should be populated with glove.840B.300d.txt file
  - http://nlp.stanford.edu/data/glove.840B.300d.zip and extract the zip file.
c. model folder should be populated with Universal Sentence Encoder transformer model V3
  - https://tfhub.dev/google/universal-sentence-encoder-large/3
  
Once all these dependencies are met, Jupyter Notebook should execute and compare all image retrieval pipelines.

The overall flow of the image retrieval comparison notebook can be shown as:

![Flow]('flowchart.png')



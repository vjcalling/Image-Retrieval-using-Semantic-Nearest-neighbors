# Image-Retrieval-using-Semantic-Nearest-neighbors #
Deep Learning approach where we use computer vision for image captioning and NLP for image retrieval using user-defined query.

# This repository is split into 4 modules: #

## a. Custom image dataset ##
  - This dataset is used for image captioning as well as image retrieval pipelines.
  
## b. Custom query dataset ##
  - A custom set of 50+ user-defined queries are provided along with the images to be retrieved mapping. This mapping is used in evaluating various image retrieval approaches.
  
## c. Image captioning pipeline ##
  - This code is used to generate image captions for the custom image dataset created as part of point a.
  
## d. Image retrieval pipelines ##
  - This code has various approaches tried for image retrieval. This has pipelines using Google's Universal Sentence Encoder and InferSent as the sentence embeddings. Various nearest neighbor techniques like Facebook's FAISS (approximate nearest neighbor) and Cosine distance are also evaulated here.

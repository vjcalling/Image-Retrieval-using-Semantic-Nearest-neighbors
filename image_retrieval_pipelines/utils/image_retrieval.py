import tensorflow as tf
import tensorflow_hub as hub
import time
import faiss
import torch
from models import InferSent
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior() 

from scipy.spatial import distance
from .sim_measures import SimMeasures
from .embeddings import Embeddings
from .nlp_preprocess import process_text
from .display_helper import find_img_idx
from .file_helper import get_captions_df, get_clean_captions

class ImageRetrieval:

    def __init__(self, emb):
        self.embedded_text = None
        self.text_input = None
        self.tf_session = None
        self.emb = emb
        self.captions_df = get_captions_df()
        self.clean_captions = get_clean_captions(self.captions_df['caption'])
        self.model = None    #required in case of Infersent
        self.index = None
        
        if emb == Embeddings.USE:
            self.init_use_model()
        else:
            self.init_infersent_model()
            
        self.CAPTION_VECTORS = self.get_captions_vectors()

    def get_captions_vectors(self):
        if self.emb == Embeddings.USE:
            return self.tf_session.run(self.embedded_text, feed_dict={self.text_input: self.clean_captions})
        else:
            return self.model.encode(self.clean_captions)


    def get_recall_list(self, expected_idx_list, received_idx_list):
        recall_score_list = []
        relevant_image_count = []
        total_queries = len(expected_idx_list)

        for i in range(total_queries):
            y_true = expected_idx_list[i]
            y_pred = received_idx_list[i]

            #recall = recall_score(y_true, y_pred, average='macro') #reason for not using this is, order can be different
            recall = len(list(set(y_true) & set(y_pred)))/len(y_true)
            
            recall_score_list.append(recall)
            relevant_image_count.append(len(y_true))
        
        return recall_score_list, relevant_image_count
        

    def semantic_search(self, query, data, vectors, embedded_text, text_input, model, tech, emb, captions_df):
        query = process_text(query)    
        if isinstance(query, str):    #if single query is passed, convert that to list.
            query = [query]
        
        if emb == Embeddings.USE:
            query_vec = model.run(embedded_text, feed_dict={text_input: query})[0].ravel()
        elif emb == Embeddings.INFERSENT:
            query_vec = model.encode(query, tokenize=True)[0].ravel()
        
        res = []
        for i, d in enumerate(data):
            qvec = vectors[i].ravel()
            if tech == SimMeasures.COSINE:
                sim = distance.cosine(query_vec, qvec)
            else:
                sim = distance.euclidean(query_vec, qvec)
                
            img_path = captions_df['image_files'][i]
            res.append((sim, d[:100], img_path, i))
            
        return sorted(res, key=lambda x : x[0], reverse=False)

    def get_matching_k_captions(self, query_str, clean_captions, CAPTION_VECTORS, embedded_text, text_input, tf_session, k, tech, emb, captions_df):
        results = self.semantic_search(query_str, clean_captions, CAPTION_VECTORS, embedded_text, text_input, self.model, tech, emb, captions_df)
        results = results[:k]
        return results



    def predict_img_indices(self,query_to_img, sim_tech):
        expected_idx_list = []
        predicted_idx_list = []
        count_row = query_to_img.shape[0] 

        if sim_tech == SimMeasures.FAISS:
            d = 4096
            index = faiss.IndexFlatL2(d)   # build the index
            index.add(self.CAPTION_VECTORS)

        for i in range(count_row):
            user_query = query_to_img['query'][i]
            expected_idx = query_to_img['img_name'][i]
            expected_idx = expected_idx.strip('][').split(',')
            expected_idx = list(map(int, expected_idx))    
            expected_idx_list.append(expected_idx)

            k = len(expected_idx)
            
            if sim_tech == SimMeasures.FAISS:
                query_vec = self.model.encode([user_query], tokenize=True)[0].ravel()
                xq = query_vec.reshape(1,4096)
                D, I = index.search(xq, k)
                predicted_idx = []
                for j in range(len(I[0])):
                    img_idx = I[0][j]
                    predicted_idx.append(find_img_idx(self.captions_df.loc[img_idx]['image_files']))
                predicted_idx_list.append(predicted_idx)
            else:
                result_captions = self.get_matching_k_captions(user_query, self.clean_captions, self.CAPTION_VECTORS, self.embedded_text, self.text_input, self.tf_session, k, sim_tech, self.emb, self.captions_df)
                predicted_idx = []
                for j in range(len(result_captions)):
                    predicted_idx.append(find_img_idx(result_captions[j][2]))
                    
                predicted_idx_list.append(predicted_idx)
        
        return expected_idx_list, predicted_idx_list

    def init_use_model(self):
        # using tensorflow-hub USE version 3 implementation. 
        # Downloaded the same to local for quick initialization. Using http URL takes ~30-40s as compared to local 5-6s.

        module_path = "./model/3/"
        start = time.time()
        g = tf.Graph()

        with g.as_default():
          self.text_input = tf.placeholder(dtype=tf.string, shape=[None])    #Using placeholder technique to pass captions from outside.
          embed = hub.Module(module_path)
          self.embedded_text = embed(self.text_input)
          init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
        g.finalize()

        # Create session and initialize.
        self.tf_session = tf.Session(graph=g)
        self.tf_session.run(init_op)
        self.model = self.tf_session

        #print("Time taken to initialize the graph: {0:.2f} seconds".format(time.time() - start))

    def init_infersent_model(self):
        model_version = 1
        MODEL_PATH = "encoder/infersent%s.pkl" % model_version
        params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                        'pool_type': 'max', 'dpout_model': 0.0, 'version': model_version}
        model = InferSent(params_model)
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # Keep it on CPU or put it on GPU
        use_cuda = False
        model = model.cuda() if use_cuda else model

        # If infersent1 -> use GloVe embeddings. If infersent2 -> use InferSent embeddings.
        W2V_PATH = 'GloVe/glove.840B.300d.txt' if model_version == 1 else 'fastText/crawl-300d-2M.vec'
        model.set_w2v_path(W2V_PATH)
        # Load embeddings of K most frequent words
        model.build_vocab_k_words(K=100000)
        self.model = model



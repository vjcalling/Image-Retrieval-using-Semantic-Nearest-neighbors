import pandas as pd
from .nlp_preprocess import process_text

CAPTIONS_FILE = './data/captions.txt'

def get_captions_df():
    # This captions.txt is created as part of Image Captioning pipeline.
    # All user images are passed to that model and it creates captions for all images. 
    # Captions are printed on the image (for quick comparision) as well as persisted to this file.

    captions_df = pd.read_csv(CAPTIONS_FILE)
    return captions_df

def get_clean_captions(raw_captions):
    clean_captions = list(map(process_text, raw_captions))
    return clean_captions

def load_query_to_img():
    query_to_img = pd.read_csv('./data/query_2_img_idx_mapping.txt', sep=';')
    return query_to_img

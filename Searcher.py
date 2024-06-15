import os
import time
import pickle
import random
import faiss
from tqdm import tqdm

class FaissSearcher:

    def __init__(self, encoder, dataset, index_name, embedding_size=768):        
        if not os.path.exists(index_name):
            index_name = index_name.replace('.index', '')
            index_name = index_name.replace('.pkl','')
            print("Encode the corpus. This might take a while")
            all_contexts = dataset 
            all_contexts_emb = encoder.encode(all_contexts, show_progress_bar=True, convert_to_numpy=True)            
            print("Store pickle file on disc")
            with open(index_name+'.pkl', 'wb') as f:
                pickle.dump({'contexts': dataset, 'embeddings': all_contexts_emb}, f)
            print("Store faiss index on disc")
            index = faiss.IndexFlatL2(embedding_size)
            index.add(all_contexts_emb)
            faiss.write_index(index, index_name+'.index')

        else:
            index_name = index_name.replace('.index', '')
            index_name = index_name.replace('.pkl','')
            print("Load Faiss from disc")
            with open(index_name+'.pkl', 'rb') as f:
                cache_data = pickle.load(f)
            all_contexts = cache_data['contexts']
            all_contexts_emb = cache_data['embeddings']
            index = faiss.read_index(index_name+'.index')

        self.index = index
        self.all_contexts = all_contexts
        self.all_contexts_emb = all_contexts_emb
import numpy as np
from numpy import dot
from numpy.linalg import norm


class HyDE(object):
    """
    Original HyDE Code from https://github.com/texttron/hyde/tree/main
    """
    def __init__(self, promptor, generator, encoder, searcher):
        self.promptor = promptor
        self.generator = generator
        self.encoder = encoder
        self.searcher = searcher

    def prompt(self, query):
        return self.promptor.build_prompt(query)

    def generate(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    def encode(self, query, hypothesis_documents):
        all_emb_c = []
        for c in [query] + hypothesis_documents:
            c_emb = self.encoder.encode(c)
            all_emb_c.append(np.array(c_emb))
        all_emb_c = np.array(all_emb_c)
        avg_emb_c = np.mean(all_emb_c, axis=0)
        hyde_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return hyde_vector

    def search(self, hyde_vector, k=10):
        hits = self.searcher.search(hyde_vector, k=k)
        return hits

    def e2e_search(self, query, k=10):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        hyde_vector = self.encode(query, hypothesis_documents)
        hits = self.searcher.search(hyde_vector, k=k)
        return hits


class HyDEQ(HyDE):
    """
    Class for HyDEQ Pipe Line.
    """
    def __init__(self, promptor, generator, encoder, searcher):
        super(HyDEQ, self).__init__(promptor, generator, encoder, searcher)

    def generate_hypothesis(self, query, n=3):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt, n)
        hypothesis_documents = [self.post_processing(hyd) for hyd in hypothesis_documents]
        return hypothesis_documents

    def generate_answer(self, query, text, n=1):
        prompt = self.promptor.build_prompt(query, text)
        hypothesis_documents = self.generator.generate(prompt, n)
        hypothesis_documents = [self.post_processing(hyd) for hyd in hypothesis_documents]
        return hypothesis_documents
    
    def encode_documents(self, hypothesis_documents):
        docs_emb_c = []
        for c in hypothesis_documents:
            c_emb = self.encoder.encode(c)
            docs_emb_c.append(np.array(c_emb))
        docs_emb_c = np.array(docs_emb_c)
        avg_emb_c = np.mean(docs_emb_c, axis=0)
        docs_vector = avg_emb_c.reshape((1, len(avg_emb_c)))
        return docs_vector

    def encode_query(self, query):
        query_emb_c = self.encoder.encode(query)
        query_vector = np.array(query_emb_c)
        return query_vector

    def rerank_by_chunk(self, hits, query):
        hit_embs = []
        for idx, hit in enumerate(hits):
            hit_emb = self.encoder.encode(hit)
            score = self.cosine_similarity(self.encode_query(query), np.array(hit_emb))
            hit_set = {'idx':idx, 'text':hit, 'vector':hit_emb, 'score':score}
            hit_embs.append(hit_set)
        reranked_embs = sorted(hit_embs, key=lambda x: x['score'], reverse=True)
        return reranked_embs

    def rerank_by_line(self, hits, query):
        hit_embs = []
        for idx, hit in enumerate(hits):
            hit_emb = self.encoder.encode(hit)
            hit_line_embs = [self.encoder.encode(line) for line in self.split_chunk_into_line(hit)]
            hit_line_scrs = [self.cosine_similarity(self.encode_query(query), np.array(emb)) for emb in hit_line_embs]
            hit_set = {'idx':idx, 'text':hit, 'vector':hit_emb, 'score':max(hit_line_scrs)}
            hit_embs.append(hit_set)
        reranked_embs = sorted(hit_embs, key=lambda x: x['score'], reverse=True)
        return reranked_embs

    def split_chunk_into_line(self, hit, tokenizer=None):
        if not tokenizer: 
            hit_line = hit.split('.')
        else:
            hit_lne = tokniezer(hit)
        return hit_line

    @staticmethod
    def cosine_similarity(a, b):
        return dot(a,b)/(norm(a)*norm(b))

    @staticmethod
    def post_processing(text):
        # for Korean Alpaca Model
        if '###답변:' in text:
            answer = text.split('###답변:')[1]
            if '<|endoftext|>' in answer:
                cleaned_answer = answer.split('<|endoftext|>')[0]
                return cleaned_answer
            else: return answer
        
        else:
            if '<|endoftext|>' in text:
                cleaned_answer = text.split('<|endoftext|>')[0] 
                return cleaned_answer
            else: return text
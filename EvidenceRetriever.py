

from sentence_transformers import SentenceTransformer
import torch
from rank_bm25 import BM25Okapi
import numpy as np
import nltk
import os
from PIL import Image
import json
import time
import io
import base64

def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class EvidenceRetriever:
    def __init__(self, sbert_model, colpali_model, colpali_processor, text_dirs, claim_image_dir, corpus_image_base_dir, batch_size):
        self.model = colpali_model
        self.processor = colpali_processor
        self.text_dirs = text_dirs
        self.claim_image_dir = claim_image_dir
        self.corpus_image_base_dir = corpus_image_base_dir
        self.bm25_k = 10
        self.top_text_results = 5
        self.top_image_results = 5
        self.pre_question_topk = 3
        self.post_question_topk = 3
        self.text_chunk_size = 8
        self.sbert_model = sbert_model

        self.batch_size = batch_size
        self._corpus_imgs = {}
        self._corpus_meta = {}
        self._corpus_embs = {}

    def get_bm25_candidates(self, query, sentences, metadata, top_k):
        k = min(top_k, len(sentences))
        docs = sentences[:k]
        metas = metadata[:k]
        tokenized = [nltk.word_tokenize(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(nltk.word_tokenize(query))
        idx = np.argsort(scores)[::-1][:k]
        return [docs[i] for i in idx], [metas[i] for i in idx]

    def generate_sbert_embs(self, texts):
        try:
            emb = self.sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        except Exception as e:
            print(e)
            emb = self.sbert_model.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=4)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        else:
            emb = np.array(emb)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        emb = emb / norms
        return emb

    def generate_colpali_embs_text(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch = self.processor.process_queries(batch_texts).to(self.model.device)
            with torch.no_grad():
                emb = self.model(**batch)
            if isinstance(emb, torch.Tensor):
                embeddings.append(emb.detach().cpu())
            else:
                try:
                    embeddings.append(emb.last_hidden_state.detach().cpu())
                except:
                    embeddings.append(torch.tensor(np.array(emb)).cpu())
            del batch
            torch.cuda.empty_cache()
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.empty((0,))

    def generate_colpali_embs_images(self, images):
        embeddings = []
        for i in range(0, len(images), self.batch_size):
            batch_imgs = images[i:i+self.batch_size]
            batch = self.processor.process_images(batch_imgs).to(self.model.device)
            with torch.no_grad():
                emb = self.model(**batch)
            if isinstance(emb, torch.Tensor):
                embeddings.append(emb.detach().cpu())
            else:
                try:
                    embeddings.append(emb.last_hidden_state.detach().cpu())
                except:
                    embeddings.append(torch.tensor(np.array(emb)).cpu())
            del batch
            torch.cuda.empty_cache()
        if embeddings:
            return torch.cat(embeddings, dim=0)
        return torch.empty((0,))

    def load_corpus_images_and_meta(self, claim_id):
        if claim_id in self._corpus_imgs:
            return self._corpus_imgs[claim_id], self._corpus_meta[claim_id]
        
        imgs = []
        meta = []
        claim_specific_image_dir = os.path.join(self.corpus_image_base_dir, str(claim_id))

        for root, dirs, files in os.walk(claim_specific_image_dir):
            for f in files:
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    path = os.path.join(root, f)
                    try:
                        img = Image.open(path).convert("RGB")
                        imgs.append(img)
                        meta.append({"path": path, "meta_root": root, "basename": os.path.basename(path)})
                    except Exception:
                        print(f"Failed to open corpus image: {path}")
                        continue
        
        self._corpus_imgs[claim_id] = imgs
        self._corpus_meta[claim_id] = meta
        print(f"Loaded {len(imgs)} corpus images for claim {claim_id} from {claim_specific_image_dir}")
        return imgs, meta

    def generate_corpus_embs(self, claim_id):
        if claim_id in self._corpus_embs:
            return self._corpus_embs[claim_id]
        
        imgs, meta = self.load_corpus_images_and_meta(claim_id)
        if len(imgs) == 0:
            self._corpus_embs[claim_id] = torch.empty((0,))
            return self._corpus_embs[claim_id]
        
        self._corpus_embs[claim_id] = self.generate_colpali_embs_images(imgs)
        print(f"Computed corpus embeddings for claim {claim_id}: {self._corpus_embs[claim_id].shape}")
        return self._corpus_embs[claim_id]

    def load_claim_images(self, claim_images, claim_id):
        objs = []
        paths = []
        _, corpus_meta = self.load_corpus_images_and_meta(claim_id)
        by_basename = {}
        for m in corpus_meta:
            by_basename.setdefault(m['basename'].lower(), []).append(m)

        for ci in claim_images:
            if isinstance(ci, str):
                candidate = os.path.join(self.claim_image_dir, ci)
                objs.append(Image.open(candidate).convert("RGB"))
                paths.append(candidate)

        print(f"Loaded {len(objs)} claim images")
        return objs, paths

    def find_similar_images(self, claim_img_objs, claim_id, top_k=3):
        imgs, meta = self.load_corpus_images_and_meta(claim_id)
        
        corpus_embs = self.generate_corpus_embs(claim_id)
        claim_embs = self.generate_colpali_embs_images(claim_img_objs)
        
        scores = self.processor.score_multi_vector(claim_embs.to(self.model.device), corpus_embs.to(self.model.device))
        mean_scores = scores.mean(dim=0).cpu().numpy()
     
        top_idx = np.argsort(mean_scores)[::-1][:min(top_k, len(mean_scores))]
        results = []
        for i in top_idx:
            results.append({"path": meta[i]["path"], "b64": image_to_base64(imgs[i]), "meta": meta[i]})
        return results

    def retrieve_images_by_relevance(self, claim_text, claim_images, claim_id):
        claim_img_objs, claim_img_paths = self.load_claim_images(claim_images, claim_id)
        pre_images = self.find_similar_images(claim_img_objs, claim_id, top_k=self.pre_question_topk)
        return {
            "relevant_images": pre_images,
            "claim_img_objs": claim_img_objs,
            "claim_img_paths": claim_img_paths,
        }

    def retrieve_post_question(self, combined_query, claim_img_objs, claim_id, top_k=3, top_text_results=10):
        sentences = []
        metadata = []
        start_time_p = time.time()
        for td in self.text_dirs:
            path = os.path.join(td, f"{claim_id}.json")
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    data = json.loads(line)
                    url2text = data.get("url2text", [])
                    url = data.get("url", "")
                    for i in range(0, len(url2text), self.text_chunk_size):
                        chunk = url2text[i:i+self.text_chunk_size]
                        concatenated = " ".join(chunk)
                        sentences.append(concatenated)
                        metadata.append({"url": url, 
                                         })
  
        bm_sentences, bm_meta = self.get_bm25_candidates(combined_query, sentences, metadata, top_k=min(self.bm25_k, len(sentences)))
        
        top_texts = []
        if len(bm_sentences) > 0:

            def get_detailed_instruct(query: str) -> str:
                task_description = 'Given a web search query, retrieve relevant passages that answer the query'
                return f'Instruct: {task_description}\nQuery: {query}'

            cand_embs = self.generate_sbert_embs(bm_sentences)
            query_emb = self.generate_sbert_embs([get_detailed_instruct(combined_query)])

            sims = (query_emb @ cand_embs.T).reshape(-1)
            
            top_idx = np.argsort(sims)[::-1][: min(top_text_results, len(sims))]
            for i in top_idx:
                top_texts.append({"text": bm_sentences[i], "meta": bm_meta[i]})

        end_time_p = time.time()
        duration_t = end_time_p - start_time_p
        
        start_time_p = time.time()
        imgs, meta = self.load_corpus_images_and_meta(claim_id)
        corpus_embs = self.generate_corpus_embs(claim_id)
        retrieved_images = []
        
        q_emb = self.generate_colpali_embs_text([combined_query])

        scores = self.processor.score_multi_vector(q_emb.to(self.model.device), corpus_embs.to(self.model.device))
        scores = scores.detach().cpu().numpy()[0]

        top_idx = np.argsort(scores)[::-1][:min(top_k, len(scores))]
        for i in top_idx:
            retrieved_images.append({"path": meta[i]["path"], "b64": image_to_base64(imgs[i]), "meta": meta[i]})

        end_time_p = time.time()
        duration_i = end_time_p - start_time_p
        
        return top_texts, retrieved_images, duration_t, duration_i

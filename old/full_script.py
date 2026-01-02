import os
import json
import io
import base64
import torch
import numpy as np
import pandas as pd
import nltk
import logging
from PIL import Image
from rank_bm25 import BM25Okapi
from colpali_engine.models import ColPali, ColPaliProcessor
from sentence_transformers import SentenceTransformer
import re
from collections import defaultdict, Counter
import time
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


nltk.download("punkt")
nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

BATCH_SIZE = 16
IMAGE_SHRINK_FACTOR = 8
#QWEN_VL_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_VL_MODEL_NAME = "./merged_model_v_2e_full_67"
COLPALI_MODEL_NAME = "vidore/colpali-v1.3"
SBERT_EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
RUN_NAME = "q3-2b-ft-colpali-emb4-fos-shrink8-URL"

if torch.cuda.is_available():
    torch_dtype_model = torch.float16
    device_map = "auto"
    device_str = "cuda"
else:
    torch_dtype_model = torch.float32
    device_map = {"": "cpu"}
    device_str = "cpu"


colpali = ColPali.from_pretrained(COLPALI_MODEL_NAME, torch_dtype=torch_dtype_model, device_map=device_map).eval()
processor = ColPaliProcessor.from_pretrained(COLPALI_MODEL_NAME)

text_sbert = SentenceTransformer(SBERT_EMBEDDING_MODEL_NAME, device=device_str)

qwen_model = Qwen3VLForConditionalGeneration.from_pretrained(
    QWEN_VL_MODEL_NAME,
    dtype=torch_dtype_model if isinstance(torch_dtype_model, torch.dtype) else None,
    device_map=device_map,
    trust_remote_code=True,
)
qwen_processor = AutoProcessor.from_pretrained(QWEN_VL_MODEL_NAME)
logger.info(f"Loaded Qwen-VL model '{QWEN_VL_MODEL_NAME}' on device {qwen_model.device}")


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class VLMCallHandler:
    def __init__(self, qwen_processor):
        self.qwen_processor = qwen_processor

    def qwen_chat_generate(self, messages, max_new_tokens=256, 
                        #  do_sample=False, 
                        #temperature=0.0
                        ):
        inputs = qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(qwen_model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        with torch.no_grad():
            # NOTE: with default do_sample=False temperature is 0 by default
            generated_ids = qwen_model.generate(**inputs,
                                                **gen_kwargs
                                                )

        input_ids = inputs.get("input_ids")
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]

        output_texts = qwen_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return [t.strip() for t in output_texts]

    def generate_question(self, claim_text, image_objs):
        prompt_text = (
            "Generate one concise essential verification question for the following claim.\n"
            "Return only the question as plain text.\n"
            f"Claim: {claim_text}"
        )

        content = []
        for img in (image_objs):
            content.append({"type": "image", "image": self.shrink_image(img, IMAGE_SHRINK_FACTOR)})

        content.append({"type": "text", "text": prompt_text})

        messages = [{"role": "user", "content": content}]

        outputs = self.qwen_chat_generate(messages, max_new_tokens=128)
        return outputs[0]


    def generate_image_answer(self, claim_text, question, image_obj):
        instruction = (
            "You are a fact-checking assistant. Your task is to analyze an image in the context of a claim and a specific question.\n"
            "1. First, determine if the provided image is relevant for answering the question about the claim.\n"
            "2. If the image is relevant, provide a concise answer to the question based *only* on what you can see in the image and the context from the claim.\n"
            "3. If the image is not relevant, state 'Image is not relevant.' as the answer.\n"
            "Respond in a JSON format with two keys: 'relevant' (boolean) and 'answer' (string).\n"
            "---\n"
            f"Claim: {claim_text}\n"
            f"Question: {question}\n"
            "---\n"
            "Here is the image to analyze:"
        )

        content = []
        content.append({"type": "image", "image": self.shrink_image(image_obj, IMAGE_SHRINK_FACTOR)})
        content.append({"type": "text", "text": instruction})

        messages = [{"role": "user", "content": content}]

        outputs = self.qwen_chat_generate(messages, max_new_tokens=512)
        raw = outputs[0]

        text = raw
        try:
            text_cleaned = text.strip().strip("`").strip()
            if text_cleaned.startswith("json"):
                text_cleaned = text_cleaned[4:].strip()
            start = text_cleaned.find("{")
            end = text_cleaned.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON object found")
            json_str = text_cleaned[start:end]
            parsed = json.loads(json_str)
            if "relevant" not in parsed or "answer" not in parsed:
                raise ValueError("Missing 'relevant' or 'answer' key")
            if not isinstance(parsed["relevant"], bool):
                parsed["relevant"] = str(parsed["relevant"]).lower() == 'true'
            return parsed
        except Exception as e:
            logger.warning(f"Failed to parse JSON response from Qwen image answer: {e}\nRaw text: {text}")
            answer_lower = text.lower()
            if "not relevant" in answer_lower or "isn't relevant" in answer_lower:
                return {"relevant": False, "answer": text}
            else:
                return {"relevant": True, "answer": text}


    def generate_verdict(self, claim_text, question, text_evidences, image_evidences):
        evidence_texts_prompt = []
        for i, e in enumerate(text_evidences):
            t = e.get("text", "")
            u = e.get("meta", {}).get("url", "")
            source = e.get("meta", {}).get("source", "") or e.get("meta", {}).get("domain", "")
            evidence_texts_prompt.append(f"[TEXT_{i}] URL:{u}\nSOURCE:{source}\nCONTENT:{t}")

        claim_images_entries = []
        evidence_images_entries = []
        for i, img_e in enumerate(image_evidences):
            meta = img_e.get("meta", {}) or {}
            src = (meta.get("source") or meta.get("meta_root") or meta.get("domain") or "").lower()
            if src == "claim":
                claim_images_entries.append((i, img_e))
            else:
                evidence_images_entries.append((i, img_e))

        claim_image_prompt_lines = []
        evidence_image_prompt_lines = []
        images_for_prompt = []

        for ci_idx, (orig_i, img_e) in enumerate(claim_images_entries):
            text = img_e.get("generated_text", "")
            path = img_e.get("path", f"claim_image_{ci_idx}")
            meta = img_e.get("meta", {}) or {}
            url = meta.get("url") or ""
            claim_image_prompt_lines.append(f"[CLAIM_IMAGE_{ci_idx}] Path: {path}\nSOURCE:claim\nURL:{url}\nDESCRIPTION: {text}")
            pil = Image.open(path).convert("RGB")
            images_for_prompt.append(self.shrink_image(pil, IMAGE_SHRINK_FACTOR))

        for ei_idx, (orig_i, img_e) in enumerate(evidence_images_entries):
            text = img_e.get("generated_text", "")
            path = img_e.get("path", f"image_{ei_idx}")
            meta = img_e.get("meta", {}) or {}
            source = meta.get("source") or meta.get("meta_root") or meta.get("domain") or ""
            url = meta.get("url") or ""
            evidence_image_prompt_lines.append(f"[IMAGE_{ei_idx}] Path: {path}\nSOURCE:{source}\nURL:{url}\nGENERATED_ANSWER: {text}")
            pil = Image.open(path).convert("RGB")
            images_for_prompt.append(self.shrink_image(pil, IMAGE_SHRINK_FACTOR))

        prompt_parts = [
            "You are a professional fact-checker. Using only the provided evidence items (text and image answers),",
            "produce exactly one JSON object and nothing else with two keys: \"verdict\" and \"justification\".",
            "",
            "VERDICT LABELS (choose exactly one):",
            " - Supported",
            " - Refuted",
            " - Not Enough Evidence",
            " - Conflicting Evidence/Cherry-picking",
            "",
            "The justification should be concise and cite evidence items by their index like [TEXT_0], [IMAGE_1],",
            "or claim images using [CLAIM_IMAGE_0] notation if you refer to the image that was provided with the claim.",
            "Do NOT hallucinate additional facts; rely only on the supplied evidence pieces.",
            "",
            "Possible reasons which can be part of the justification",
            "question_type: Text-related, Image-related, Metadata-related, Commonsense-related. \nanswer_type: Abstractive, Extractive, Unanswerable, Boolean, Image\nfact_checking_strategies: Written Evidence, Consultation, Keyword Search, Numerical Comparison, Reverse Image Search, Fact-checker Reference, Media Source Discovery, Image Analysis, Geolocation, Video Analysis, Satirical Source Identification, Audio Analysis\nrefuting_reasons: Misuse of images, Textual refuted, Others\nimage_misuse_types: Out-of-context, Others, ",
            "",
            "IMPORTANT: Images that are part of the claim are listed under [CLAIM_IMAGE_i] and are part of the claim context --",
            "they are NOT to be treated as retrieved evidence. Other images are evidence and are listed under [IMAGE_i].",
            "When actual image files are attached to this prompt, they are provided in the same order as shown below:",
            "  first: all CLAIM images ([CLAIM_IMAGE_0], [CLAIM_IMAGE_1], ...),",
            "  then: all evidence images ([IMAGE_0], [IMAGE_1], ...).",
            "",
            f"Claim: {claim_text}",
            f"Question: {question}",
            "",
            "Claim images (indexed):",
            "\n".join(claim_image_prompt_lines) if claim_image_prompt_lines else "No claim images provided.",
            "",
            "Text evidence items (indexed):",
            "\n".join(evidence_texts_prompt) if evidence_texts_prompt else "No text evidence provided.",
            "",
            "Image evidence items (indexed):",
            "\n".join(evidence_image_prompt_lines) if evidence_image_prompt_lines else "No image evidence provided.",
            "",
            "Return only the JSON object. Example:",
            '{"verdict": "Supported", "justification": "Because [TEXT_0] shows ... and [IMAGE_1] corroborates ..."}'
        ]

        full_instruction = "\n\n".join(prompt_parts)

        content = []
        for img in images_for_prompt:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": full_instruction})
        messages = [{"role": "user", "content": content}]

        outputs = self.qwen_chat_generate(messages, max_new_tokens=512, 
                                    #do_sample=False
                                    )
        raw = outputs[0]


        text = raw
        parsed = None
        try:
            text_cleaned = text.strip().strip("`").strip()
            if text_cleaned.startswith("json"):
                text_cleaned = text_cleaned[4:].strip()
            start = text_cleaned.find("{")
            end = text_cleaned.rfind("}") + 1
            if start != -1 and end != 0:
                json_str = text_cleaned[start:end]
                parsed = json.loads(json_str)
            else:
                m = re.search(r"(\{.*\})", text, flags=re.S)
                if m:
                    parsed = json.loads(m.group(1))
        except Exception as e:
            logger.warning(f"Failed to parse JSON response from Qwen verdict: {e}\nRaw text: {text}")

        if not parsed:
            logger.warning("Could not parse model output into JSON verdict. Returning Not Enough Evidence with raw output.")
            return "Not Enough Evidence", text

        raw_verdict = parsed.get("verdict") or parsed.get("label") or parsed.get("decision")
        justification = parsed.get("justification") or parsed.get("reason") or parsed.get("explanation") or ""

        normalized = self.normalize_verdict_raw(raw_verdict)

        return normalized, justification
    
    def normalize_verdict_raw(self, verdict):
        if "supporte" in verdict.lower():
            return "Supported"
        elif "refute" in verdict.lower():
            return "Refuted"
        elif "cherry" in verdict.lower():
            return "Conflicting Evidence/Cherry-picking"
        else:
            return "Not Enough Evidence"
        
    def shrink_image(self, img, factor=IMAGE_SHRINK_FACTOR):
        w, h = img.size
        nw = max(1, w // factor)
        nh = max(1, h // factor)
        return img.resize((nw, nh), resample=Image.LANCZOS)


class EvidenceRetriever:
    def __init__(self, colpali_model, colpali_processor, text_dirs, claim_image_dir, corpus_image_base_dir):
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

        self.batch_size = BATCH_SIZE
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
            emb = text_sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=32)
        except Exception as e:
            print(e)
            emb = text_sbert.encode(texts, convert_to_tensor=True, show_progress_bar=False, batch_size=4)
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
                        logger.warning(f"Failed to open corpus image: {path}")
                        continue
        
        self._corpus_imgs[claim_id] = imgs
        self._corpus_meta[claim_id] = meta
        logger.info(f"Loaded {len(imgs)} corpus images for claim {claim_id} from {claim_specific_image_dir}")
        return imgs, meta

    def generate_corpus_embs(self, claim_id):
        if claim_id in self._corpus_embs:
            return self._corpus_embs[claim_id]
        
        imgs, meta = self.load_corpus_images_and_meta(claim_id)
        if len(imgs) == 0:
            self._corpus_embs[claim_id] = torch.empty((0,))
            return self._corpus_embs[claim_id]
        
        self._corpus_embs[claim_id] = self.generate_colpali_embs_images(imgs)
        logger.info(f"Computed corpus embeddings for claim {claim_id}: {self._corpus_embs[claim_id].shape}")
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

        logger.info(f"Loaded {len(objs)} claim images")
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

t_sim = []
t_qg = []
t_tr = []
t_mm = []
t_rc = []
t_vg = []

def main():
    start_time = time.time()
    val_path = "val.json"
    text_dirs = [
        "converted_datastore/text_related/image_related_store_text_val",
        "converted_datastore/text_related/text_related_store_text_val",
    ]
    claim_image_dir = "images"
    corpus_image_base_dir = "converted_datastore/image_related/image_related_store_image_val"
    
    retriever = EvidenceRetriever(colpali, processor, text_dirs, claim_image_dir, corpus_image_base_dir)
    vlm_call_handler = VLMCallHandler(qwen_processor)

    submission_rows = []
    total = 0
    correct = 0
    
    label_totals = defaultdict(int)
    label_correct = defaultdict(int)
    pred_counts_by_gold = defaultdict(Counter)

    with open(val_path, "r", encoding="utf-8") as vf:
        data_list = json.load(vf)
        for item in data_list:
            claim_id = total
            total += 1
            claim_text = item.get("claim_text", "")
            claim_images = item.get("claim_images", item.get("claim_image", [])) or []
            gold = item.get("label", "")
            gold_norm = str(gold).strip().lower()

            start_time_p = time.time()
            relevant_image_data = retriever.retrieve_images_by_relevance(claim_text, claim_images, str(claim_id))
            relevant_images = relevant_image_data.get("relevant_images", [])
            claim_img_objs = relevant_image_data.get("claim_img_objs", [])
            claim_img_paths = relevant_image_data.get("claim_img_paths", [])
            end_time_p = time.time()
            duration_p = end_time_p - start_time_p
            t_sim.append(f"{duration_p:.4f}")

            print(f"=== Claim {claim_id} ===")
            print(f"Claim text: {claim_text}")
            print(f"Loaded claim image paths: {claim_img_paths}")
            print(f"isually relevant images (paths): {[p['path'] for p in relevant_images]}")

            q_image_objs_for_prompt = []
            q_image_objs_for_prompt.extend(claim_img_objs)
            for p in relevant_images:
                q_image_objs_for_prompt.append(Image.open(p['path']).convert('RGB'))
       

            start_time_p = time.time()
            q = vlm_call_handler.generate_question(claim_text, q_image_objs_for_prompt)
            end_time_p = time.time()
            duration_p = end_time_p - start_time_p
            t_qg.append(f"{duration_p:.4f}")
            
            questions_list = [q]
            print(f"Generated question: {q}")

            combined_query = f"{claim_text} {q}"
            top_texts, top_images, d_t, d_i = retriever.retrieve_post_question(
                combined_query, 
                claim_img_objs, 
                str(claim_id), 
                top_k=retriever.post_question_topk, 
                top_text_results=retriever.top_text_results
            )
            t_tr.append(f"{d_t:.4f}")
            t_mm.append(f"{d_i:.4f}")
            
            print(f"Top text evidence count: {len(top_texts)}")
            print(f"Top post-question images (paths): {[ti['path'] for ti in top_images]}")

            evidence_for_json = []
            final_verdict_texts = []
            final_verdict_images = []
            processed_image_paths = set()

            for tt in top_texts:
                evidence_for_json.append({"text": tt.get("text", ""), "images": [], "url": tt["meta"]["url"]})
                final_verdict_texts.append(tt)
            
            all_image_sources = []
            
            for path, img_obj in zip(claim_img_paths, claim_img_objs):
                all_image_sources.append({
                    "path": path,
                    "img_obj": img_obj,
                    "b64": image_to_base64(img_obj),
                    "meta": {"path": path, "source": "claim"}
                })
            
            for p in relevant_images:
                meta = dict(p.get('meta', {}))
                meta['source'] = 'pre'
                all_image_sources.append({
                    "path": p['path'],
                    "img_obj": Image.open(p['path']).convert("RGB"),
                    "b64": p['b64'],
                    "meta": meta
                })
            
            for ti in top_images:
                meta = dict(ti.get('meta', {}))
                meta['source'] = 'retrieved'
                all_image_sources.append({
                    "path": ti['path'],
                    "img_obj": Image.open(ti['path']).convert("RGB"),
                    "b64": ti['b64'],
                    "meta": meta
                })

            start_time_p = time.time()
            for img_source in all_image_sources:
                path = img_source.get("path")
                if not path or path in processed_image_paths:
                    continue
            
                processed_image_paths.add(path)
            
                img_obj = img_source.get("img_obj")
        
                print(f"Generating answer for image: {path}")
                img_answer_result = vlm_call_handler.generate_image_answer(claim_text, q, img_obj)
            
                is_claim_image = img_source.get("meta", {}).get("source") == "claim"
                include_image = bool(img_answer_result.get("relevant")) or is_claim_image
            
                generated_text = img_answer_result.get("answer", "").strip()
            
                if include_image:
                    b64 = img_source.get("b64")
 
                    evidence_for_json.append({
                        "text": generated_text,
                        "images": [b64],
                        "path": path
                    })
            
                    final_verdict_images.append({
                        "path": path,
                        "b64": b64,
                        "meta": img_source.get("meta"),
                        "generated_text": generated_text
                    })
                else:
                    print(f"  -> Not relevant (and not a claim image).")

            end_time_p = time.time()
            duration_p = end_time_p - start_time_p
            t_rc.append(f"{duration_p:.4f}")


            start_time_p = time.time()
            verdict, justification = vlm_call_handler.generate_verdict(
                claim_text, 
                q, 
                final_verdict_texts, 
                final_verdict_images
            )
            end_time_p = time.time()
            duration_p = end_time_p - start_time_p
            t_vg.append(f"{duration_p:.4f}")

            submission_rows.append({
                "id": claim_id,
                "questions": questions_list,
                "evidence": evidence_for_json,
                "verdict": verdict,
                "justification": justification,
            })

            pred_norm = str(verdict).strip().lower()
            label_totals[gold_norm] += 1
            pred_counts_by_gold[gold_norm][pred_norm] += 1
            if pred_norm == gold_norm:
                label_correct[gold_norm] += 1
                correct += 1

            print(f"Claim {claim_id} predicted: {verdict} | gold: {gold} | correct: {pred_norm == gold_norm}")
                
            with open(f"submission_{RUN_NAME}.json", "w", encoding="utf-8") as f:
                json.dump(submission_rows, f, indent=2, ensure_ascii=False)

    end_time = time.time()
    duration = end_time - start_time
    
    with open(f"submission_{RUN_NAME}.json", "w", encoding="utf-8") as f:
        json.dump(submission_rows, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved results to submission_{RUN_NAME}.json")

    accuracy = correct / total if total > 0 else 0.0
    print(f"Total: {total} Correct: {correct} Accuracy: {accuracy:.4f}")

    print("Per-label accuracy:")
    for label in sorted(label_totals.keys()):
        tot = label_totals[label]
        corr = label_correct.get(label, 0)
        acc = corr / tot if tot > 0 else 0.0
        print(f"  Label '{label}': {corr}/{tot} correct -> accuracy={acc:.4f}")

    all_pred_labels = set()
    for counts in pred_counts_by_gold.values():
        all_pred_labels.update(counts.keys())
    all_pred_labels = sorted(all_pred_labels)

    conf_rows = []
    for gold_label in sorted(pred_counts_by_gold.keys()):
        row = {"gold_label": gold_label}
        for p in all_pred_labels:
            row[p] = pred_counts_by_gold[gold_label].get(p, 0)
        conf_rows.append(row)

    conf_df = pd.DataFrame(conf_rows).fillna(0).astype({c: int for c in conf_rows[0].keys() if c != "gold_label"}) if conf_rows else pd.DataFrame()
    print("Confusion matrix (rows=gold labels, columns=predicted labels):")
    print(conf_df.to_string(index=False))

    with open(f"results_{RUN_NAME}.txt", "w") as f:
        accuracy = correct / total if total > 0 else 0.0
        print(f"Total: {total} Correct: {correct} Accuracy: {accuracy:.4f}", file=f)
    
        print("\nPer-label accuracy:", file=f)
        for label in sorted(label_totals.keys()):
            tot = label_totals[label]
            corr = label_correct.get(label, 0)
            acc = corr / tot if tot > 0 else 0.0
            print(f"  Label '{label}': {corr}/{tot} correct -> accuracy={acc:.4f}", file=f)
    
        all_pred_labels = set()
        for counts in pred_counts_by_gold.values():
            all_pred_labels.update(counts.keys())
        all_pred_labels = sorted(all_pred_labels)
    
        conf_rows = []
        for gold_label in sorted(pred_counts_by_gold.keys()):
            row = {"gold_label": gold_label}
            for p in all_pred_labels:
                row[p] = pred_counts_by_gold[gold_label].get(p, 0)
            conf_rows.append(row)
    
        conf_df = pd.DataFrame(conf_rows).fillna(0).astype({c: int for c in conf_rows[0].keys() if c != "gold_label"}) if conf_rows else pd.DataFrame()
        
        print("\nConfusion matrix (rows=gold labels, columns=predicted labels):", file=f)
        print(conf_df.to_string(index=False), file=f)

        print(f"Elapsed time - total: {duration:.4f}s", file=f)

        durations = np.array(t_sim, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_sim: {average:.4f}s", file=f)
        durations = np.array(t_qg, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_qg: {average:.4f}s", file=f)
        durations = np.array(t_tr, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_tr: {average:.4f}s", file=f)
        durations = np.array(t_mm, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_mm: {average:.4f}s", file=f)
        durations = np.array(t_rc, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_rc: {average:.4f}s", file=f)
        durations = np.array(t_vg, dtype=float)
        average = np.mean(durations)
        print(f"Elapsed time - mean t_vg: {average:.4f}s", file=f)

if __name__ == "__main__":
    main()

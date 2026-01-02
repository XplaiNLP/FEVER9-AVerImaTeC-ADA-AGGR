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
from EvidenceRetriever import EvidenceRetriever
from VLMCallHandler import VLMCallHandler

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
    
    retriever = EvidenceRetriever(text_sbert, colpali, processor, text_dirs, claim_image_dir, corpus_image_base_dir, BATCH_SIZE)
    vlm_call_handler = VLMCallHandler(qwen_model, qwen_processor, BATCH_SIZE)

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

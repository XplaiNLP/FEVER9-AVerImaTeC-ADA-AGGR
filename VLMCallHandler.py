import os
import json
import io
import base64
import torch
from PIL import Image
import re



def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class VLMCallHandler:
    def __init__(self, qwen_model, qwen_processor, shrink_factor):
        self.qwen_model = qwen_model
        self.qwen_processor = qwen_processor
        self.shrink_factor = shrink_factor

    def qwen_chat_generate(self, messages, max_new_tokens=256, 
                        #  do_sample=False, 
                        #temperature=0.0
                        ):
        inputs = self.qwen_processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.qwen_model.device)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)

        with torch.no_grad():
            # NOTE: with default do_sample=False temperature is 0 by default
            generated_ids = self.qwen_model.generate(**inputs,
                                                **gen_kwargs
                                                )

        input_ids = inputs.get("input_ids")
        generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, generated_ids)]

        output_texts = self.qwen_processor.batch_decode(
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
            content.append({"type": "image", "image": self.shrink_image(img)})

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
        content.append({"type": "image", "image": self.shrink_image(image_obj)})
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
            print(f"Failed to parse JSON response from Qwen image answer: {e}\nRaw text: {text}")
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
            images_for_prompt.append(self.shrink_image(pil))

        for ei_idx, (orig_i, img_e) in enumerate(evidence_images_entries):
            text = img_e.get("generated_text", "")
            path = img_e.get("path", f"image_{ei_idx}")
            meta = img_e.get("meta", {}) or {}
            source = meta.get("source") or meta.get("meta_root") or meta.get("domain") or ""
            url = meta.get("url") or ""
            evidence_image_prompt_lines.append(f"[IMAGE_{ei_idx}] Path: {path}\nSOURCE:{source}\nURL:{url}\nGENERATED_ANSWER: {text}")
            pil = Image.open(path).convert("RGB")
            images_for_prompt.append(self.shrink_image(pil))

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
            print(f"Failed to parse JSON response from Qwen verdict: {e}\nRaw text: {text}")

        if not parsed:
            print("Could not parse model output into JSON verdict. Returning Not Enough Evidence with raw output.")
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
        
    def shrink_image(self, img):
        w, h = img.size
        nw = max(1, w // self.shrink_factor)
        nh = max(1, h // self.shrink_factor)
        return img.resize((nw, nh), resample=Image.LANCZOS)


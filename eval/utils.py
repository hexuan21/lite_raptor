import os
import json
import requests
from openai import OpenAI
import time
import re
from collections import defaultdict

QA_MAX_TRY=3

HOTPOTQA_PREFIX="You're a helpful chat assistant. \nBased on your knowledge and the context (if provided), answer the question based on your knowledge and the context. Firstly try extracing substring from the context, if it's hard, generate a short answer yourself. \n\nOnly output the short answer to this question, DO NOT include anything else before or after your answer." 

def _llm_openrouter_api(
    user_prompt,
    model_name="gpt-4o-mini",
    chat_config={},
    ):
    api_key=os.environ["OPENROUTER_API_KEY"]
    llm_base_url=os.environ["OPENROUTER_BASE_URL"]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    url = f"{llm_base_url}/chat/completions"

    messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                ]
            }
        ]
        
    payload = {
        "model": model_name,
        "messages": messages, 
        "max_tokens": chat_config.get("max_tokens", 1024),
        "temperature": chat_config.get("temperature", 1.0)
    }
    thinking_enabled=chat_config.get("thinking_enabled",False)
    if thinking_enabled == True:
        payload["reasoning"]={
            "exclude": False,
            "max_tokens": chat_config.get("thinking_budget", 2048)
        }
        payload["max_tokens"] = chat_config.get("max_tokens", 1024) + chat_config.get("thinking_budget", 2048)
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        # print(f"[INFO] Open Router response status code: {response.status_code}")
        if response.status_code != 200:
            print(f"[ERR] Open Router API error: {response.text}")
            return ""
        
        thinking = str(response.json()['choices'][0]['message'].get('reasoning', ''))
        output = str(response.json()['choices'][0]['message'].get('content', ''))
        
        if thinking_enabled:
            res = "<think>"+thinking+"</think>"+"\n"+output
        else:
            res = output
        print("currently model's answer:", res)
        return res
    except Exception as e:
        print(f"[ERR] error {e}")
        return ""



# ----------------------------
# (EM / F1) metrics
# ----------------------------
def _normalize_answer(s: str) -> str:
    import string
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def metric_f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize_answer(pred).split()
    gold_tokens = _normalize_answer(gold).split()
    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 0.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0
    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            num_same += 1
            common[t] -= 1
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)

def metric_exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize_answer(pred) == _normalize_answer(gold) else 0.0


# ----------------------------
# Gold Context Extraction for oracle setting
# ----------------------------
def extract_gold_context(qa_item, sep="\n"):
    title2sents = {title: sents for title, sents in qa_item["context"]}
    
    idxs = defaultdict(set)
    for title, sent_idx in qa_item["supporting_facts"]:
        idxs[title].add(sent_idx)

    out = []
    for title, sents in qa_item["context"]:
        if title not in idxs:
            continue
        for i in sorted(idxs[title]):
            if 0 <= i < len(sents):
                out.append(sents[i])
    return out, sep.join(out)
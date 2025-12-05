import json
import os
from tqdm import tqdm
from utils import _llm_azure_api,_llm_openrouter_api,MAX_TRY
import numpy as np
import string


def llm_acc_check(judge_model_name,res_path,use_azure,):
    res_list=json.load(open(res_path,"r"))
    judge_list=[]
    judge_path=res_path.replace(".json","_judge.json")
    
    correct_num=0
    total_num=len(res_list)
    for res_item in tqdm(res_list):
        id=res_item["id"]
        q=res_item["question"]
        ref=res_item["ref_answer"]
        ans=res_item["model_answer"]
        prompt=f"You are a helpful chat assistant. Please check and determine if the student's answer to the question is correct according to the reference answer. \nHere's the reference answer: {ref}. \nHere's the student's answer: {ans}. If correct, output '1'; else, output '0'. DO NOT include anything else before or after your answer."
        
        try_counts=0
        while True:
            if try_counts>=MAX_TRY:
                break
            if use_azure:
                res=_llm_azure_api(prompt,judge_model_name)
            else:
                res=_llm_openrouter_api(prompt,judge_model_name)
            try_counts+=1
            if len(res)>0:
                break
            print(f"trying again, {try_counts}-try (max:{MAX_TRY} times)")
            
        judge_list.append({
            "id":id,
            "question":q,
            "ref_answer":ref,
            "model_answer":ans,
            "judge":res,
        })
        if "1" in res:
            correct_num+=1
        
    print(f"Accuracy: {correct_num/total_num}({correct_num}/{total_num})")
        
    with open(judge_path,"w") as f:
        json.dump(judge_list,f,indent=4)






def normalize_text(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = " ".join(s.split())
    return s


def exact_match_score(res_path) -> float:
    res_list=json.load(open(res_path,"r"))
    gold_answers,model_answers=[x["ref_answer"] for x in res_list],[x["model_answer"] for x in res_list]
    
    assert len(gold_answers) == len(model_answers)

    total = len(gold_answers)
    if total == 0:
        return 0.0

    em_count = 0
    for ref, pred in zip(gold_answers, model_answers):
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)
        if ref_norm == pred_norm:
            em_count += 1

    score = em_count / total
    print(f"Exact Match:", round(score, 4))
    return round(score, 4)


def f1_score(res_path) -> float:
    res_list=json.load(open(res_path,"r"))
    gold_answers,model_answers=[x["ref_answer"] for x in res_list],[x["model_answer"] for x in res_list]
    
    assert len(gold_answers) == len(model_answers)

    total = len(gold_answers)
    if total == 0:
        return 0.0

    def f1_single(ref: str, pred: str) -> float:
        ref_norm = normalize_text(ref)
        pred_norm = normalize_text(pred)

        ref_tokens = ref_norm.split()
        pred_tokens = pred_norm.split()

        if len(ref_tokens) == 0 and len(pred_tokens) == 0:
            return 1.0
        if len(ref_tokens) == 0 or len(pred_tokens) == 0:
            return 0.0

        ref_counter = {}
        for t in ref_tokens:
            ref_counter[t] = ref_counter.get(t, 0) + 1

        common = 0
        for t in pred_tokens:
            if ref_counter.get(t, 0) > 0:
                common += 1
                ref_counter[t] -= 1

        if common == 0:
            return 0.0

        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    f1_sum = 0.0
    for ref, pred in zip(gold_answers, model_answers):
        f1_sum += f1_single(ref, pred)

    score = f1_sum / total
    print(f"F1 Score:", round(score, 4))
    return round(score, 4)

    
    
if __name__ == "__main__":
    res_dir="eval/res"
    
    # res_path=f"{res_dir}/hotpotqa_res_bsline_llm_gpt-4o-mini_wo_ctx.json"
    # res_path=f"{res_dir}/hotpotqa_res_bsline_dpr.json"
    # res_path=f"{res_dir}/hotpotqa_res_bsline_bm25.json"
    # res_path=f"{res_dir}/hotpotqa_res.json"
    # res_path=f"{res_dir}/hotpotqa_res_root_only_start3_1layers.json"
    res_path=f"{res_dir}/hotpotqa_res_leaf_only_start0_1layers.json"
    # res_path=f"{res_dir}/hotpotqa_res_hier_start3_4layers.json"
    exact_match_score(res_path)
    f1_score(res_path)
    
    # use_azure=0
    # judge_model_name="gpt-4o-mini"
    # res_path=f"{res_dir}/hotpotqa_gpt-4o.json"
    # llm_acc_check(udge_model_name,res_path,use_azure,)
    
    # method              em     f1
    # gpt-4o-mini         20.30  33.01
    # bm25                35.90  48.90
    # dpr                 36.20  50.45
    # raptor (collapsed)  33.60  51.13
    # raptor (root only)  14.80  20.50
    # raptor (leaf only)  39.40  52.17
    # raptor (hier)       22.50  31.44
    # hipporag2           63.00  77.60 
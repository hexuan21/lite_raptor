import os
import json
from tqdm import tqdm
import argparse
import random

from utils import _llm_openrouter_api,metric_exact_match,metric_f1_score,extract_gold_context
from utils import QA_MAX_TRY,HOTPOTQA_PREFIX



def llm_run_qa(model_name,context_type,use_azure,):
    
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    all_corpus="\n\n".join(docs)

    
    data=json.load(open(EVAL_DATA_PATH,"r"))
    
    ids=[x['_id'] for x in data]
    questions=[x['question'] for x in data]
    answers=[x['answer'] for x in data]
    all_contexts=[]
    for item in data:
        context_str=""
        for x in item["context"]:
            context_str+="\nCaption:"+x[0]+"\nContent:\n"+"".join(x[1])
        all_contexts.append(context_str)
    
    
    res_list=[] 
    for id,q,a,ctx in tqdm(zip(ids,questions,answers,all_contexts)):
        prefix=HOTPOTQA_PREFIX
        if context_type=="no_context":
            prompt=prefix+f"\n\nQuestion: {q}. "
        elif context_type=="ten_psg":
            prompt=prefix+f"\n\nContext:{ctx}.\n\nQuestion: {q}. "
        elif context_type=="all_corpus":
            prompt=prefix+f"\n\nCorpus:{all_corpus}.\n\nQuestion: {q}. "
        else:
            raise ValueError(f"Invalid context_type: {context_type}")
            
        try_counts=0
        while True:
            if try_counts>=QA_MAX_TRY:
                break
            res=_llm_openrouter_api(prompt,model_name)
                
            try_counts+=1
            
            if len(res)>0:
                break
            
            print(f"trying again, {try_counts}-try (max:{QA_MAX_TRY} times)")
            
        print("\n"+a)
        print(res)            
        res_list.append({
            "id":id,
            "question":q,
            "ref_answer":a,
            "model_answer":res,
        })
    
    if context_type=="no_context":
        res_path=f"{RES_DIR}/hotpotqa_res_bsline_llm_{model_name}_wo_ctx.json"
    elif context_type=="ten_psg":
        res_path=f"{RES_DIR}/hotpotqa_res_bsline_llm_{model_name}.json"
    elif context_type=="all_corpus":
        res_path=f"{RES_DIR}/hotpotqa_res_bsline_llm_{model_name}_all_corpus.json"
    else:
        raise ValueError(f"Invalid context_type: {context_type}")
    
    with open(res_path,"w") as f:
        json.dump(res_list,f,indent=4)
        

def llm_run_qa(model_name,context_type,max_qa_items):
    
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)
    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]
    all_corpus="\n\n".join(docs)

    
    data=json.load(open(EVAL_DATA_PATH,"r"))[:max_qa_items]
    all_contexts=[]
    for item in data:
        context_str=""
        for x in item["context"]:
            context_str+="\nCaption:"+x[0]+"\nContent:\n"+"".join(x[1])
        all_contexts.append(context_str)
    
    name_base=f"vanilla_{model_name}"
    
    if context_type=="no_context":
        name_base+="_no_ctx"
    elif context_type=="gold_context":
        name_base+="_gold_ctx"
    elif context_type=="random_ten_psg":
        name_base+="_random_ten_psg"
    elif context_type=="ten_psg":
        name_base+="_ten_psg"
    else:
        raise ValueError(f"Invalid context_type: {context_type}")
    
    res_path=f"{RES_DIR}/pred_{name_base}.json"
    metric_path = f"{RES_DIR}/metric_{name_base}.json"
    
    pred_list=[] 
    total_em=0
    total_f1=0
    n=0
    for idx,item in tqdm(enumerate(data)):
        id=item['_id']
        q=item['question']
        gold=item['answer']
        
        prefix=HOTPOTQA_PREFIX
        if context_type=="no_context":
            prompt=prefix+f"\n\nQuestion: {q}. "
        
        elif context_type=="gold_context":
            gold_ctx=extract_gold_context(item)
            prompt=prefix+f"\n\nContext:{gold_ctx}.\n\nQuestion: {q}. "
        
        elif context_type=="random_ten_psg":
            random_ten_psg=random.choice(all_contexts)
            prompt=prefix+f"\n\nCorpus:{random_ten_psg}.\n\nQuestion: {q}. "
        
        elif context_type=="ten_psg":
            ten_psg=""
            for x in item["context"]:
                ten_psg+="\nCaption:"+x[0]+"\nContent:\n"+"".join(x[1])
            prompt=prefix+f"\n\nContext:{ten_psg}.\n\nQuestion: {q}. "
        else:
            raise ValueError(f"Invalid context_type: {context_type}")
            
        try_counts=0
        while True:
            if try_counts>=QA_MAX_TRY:
                break
 
            pred=_llm_openrouter_api(prompt,model_name)
                
            try_counts+=1
            
            if len(pred)>0:
                break
            
            print(f"trying again, {try_counts}-try (max:{QA_MAX_TRY} times)")
        
        em = metric_exact_match(pred, gold) if gold else 0.0
        f1 = metric_f1_score(pred, gold) if gold else 0.0
        total_em += em
        total_f1 += f1
        n+=1
        print(f"\nquestion : {q}")
        print(f"gold_answer : {gold}")
        print(f"model_answer : {pred}")
        print(f"EM={em:.3f}, F1={f1:.3f}")          
        pred_list.append({
            "id":id,
            "question":q,
            "gold":gold,
            "pred":pred,
        })
        with open(res_path,"w") as f:
            json.dump(pred_list,f,indent=4)
            
    avg_em=total_em / n if n else 0.0
    avg_f1=total_f1 / n if n else 0.0
    metrics = {
        "count": n,
        "em": avg_em,
        "f1": avg_f1,
        "context_type": context_type,
        "model_name": model_name,
        "max_qa_items": max_qa_items,
        "res_path": res_path,
        "metric_path": metric_path,
    }
        
    with open(metric_path,"w") as f:
        json.dump(metrics,f,indent=4)

    print(f"[Done] [{n}/{len(data)}] EM={avg_em:.4f} F1={avg_f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing HippoRAG")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--context_type',type=str, default="no_context")
    parser.add_argument('--max_qa_items',type=int, default=100)
    args = parser.parse_args()
    
    CORPUS_PATH = f"data/hotpotqa_corpus.json"
    EVAL_DATA_PATH="data/hotpotqa_1000.json"
    
    RES_DIR="eval/res"
    os.makedirs(RES_DIR,exist_ok=True)
    
    model_name=args.model_name
    context_type=args.context_type
    max_qa_items=args.max_qa_items
    llm_run_qa(model_name,context_type,max_qa_items)

    # python eval/eval_vanilla_llm.py --model_name "gpt-4o-mini" --context_type "no_context"
    
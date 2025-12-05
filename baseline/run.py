import os
import json
from tqdm import tqdm
import argparse

from utils import _llm_azure_api,_llm_openrouter_api,MAX_TRY



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
        prefix="You're a helpful chat assistant. \nBased on your knowledge and the context (if provided), answer the question based on your knowledge and the context. Firstly try extracing substring from the context, if it's hard, generate a short answer yourself. \n\nOnly output the short answer to this question, DO NOT include anything else before or after your answer."
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
            if try_counts>=MAX_TRY:
                break
            
            if use_azure:
                res=_llm_azure_api(prompt,model_name)
            else:
                res=_llm_openrouter_api(prompt,model_name)
            try_counts+=1
            if len(res)>0:
                break
            
            print(f"trying again, {try_counts}-try (max:{MAX_TRY} times)")
            
        print("\n"+a)
        print(res)            
        res_list.append({
            "id":id,
            "question":q,
            "ref_answer":a,
            "model_answer":res,
        })
    
    if context_type=="no_context":
        res_path=f"{res_dir}/hotpotqa_{model_name}_wo_ctx.json"
    elif context_type=="ten_psg":
        res_path=f"{res_dir}/hotpotqa_{model_name}.json"
    elif context_type=="all_corpus":
        res_path=f"{res_dir}/hotpotqa_{model_name}_all_corpus.json"
    else:
        raise ValueError(f"Invalid context_type: {context_type}")
    
    with open(res_path,"w") as f:
        json.dump(res_list,f,indent=4)
        
    
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing HippoRAG")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    parser.add_argument('--context_type',type=str, default="no_context")
    parser.add_argument('--use_azure',action='store_true')
    args = parser.parse_args()
    
    CORPUS_PATH = f"eval_data/hotpotqa_corpus.json"
    EVAL_DATA_PATH="eval_data/hotpotqa.json"
    
    res_dir="baseline/res"
    os.makedirs(res_dir,exist_ok=True)
    
    model_name=args.model_name
    context_type=args.context_type
    use_azure=args.use_azure
    llm_run_qa(model_name,context_type,use_azure,)
    
    # python baseline/run.py --context_type "ten_psg" --model_name "gpt-4o-mini"
    # python baseline/run.py --context_type "no_context" --model_name "gpt-4o-mini"
    
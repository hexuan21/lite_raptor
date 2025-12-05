import os
import json
from tqdm import tqdm
import argparse
from raptor import RetrievalAugmentation
from datetime import datetime


def run_raptor(bench):
    if bench=="hotpotqa":
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
        
        all_context_one_str="\n\n".join(all_contexts)
        print(len(all_context_one_str))
    
    RA=None
    if not os.path.exists(TREE_SAVE_PATH):
        RA = RetrievalAugmentation()
        RA.add_documents(all_context_one_str)
        RA.save(TREE_SAVE_PATH)
    else:
        RA = RetrievalAugmentation(tree=TREE_SAVE_PATH)
    
    res_list=[]

    prefix="You're a helpful chat assistant. \nBased on your knowledge and the context (if provided), answer the question based on your knowledge and the context. Firstly try extracing substring from the context, if it's hard, generate a short answer yourself. \n\nOnly output the short answer to this question. If you can't answer the question, only output a empty string ''. DO NOT include anything else before or after your answer."     
    for id,q,ref_answer in tqdm(zip(ids,questions,answers)):
        q=prefix+q
        model_answer = RA.answer_question(question=q,start_layer=START_LAYER,num_layers=NUM_LAYERS,collapse_tree=COLLAPSE_TREE)

        print("\n"+model_answer)
        print(ref_answer)            
        res_list.append({
            "id":id,
            "question":q,
            "ref_answer":ref_answer,
            "model_answer":model_answer,
        })
    if START_LAYER is None and NUM_LAYERS is None:
        res_path=f"{RES_DIR}/{bench}_res.json"
    else:
        res_path=f"{RES_DIR}/{bench}_res_{ablation_name}_start{START_LAYER}_{NUM_LAYERS}layers.json"
    with open(res_path,"w") as f:
        json.dump(res_list,f,indent=4)
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, default="hotpotqa")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    bench=args.bench
    if bench=="hotpotqa":
        EVAL_DATA_PATH="data/hotpotqa_1000.json"
    else:
        EVAL_DATA_PATH=None
    
    RES_DIR="eval/res"
    TREE_SAVE_PATH = f"eval/tree/{bench}_tree"
    os.makedirs(RES_DIR,exist_ok=True)
    os.makedirs("eval/tree",exist_ok=True)
    
    ablation_name=None
    COLLAPSE_TREE=True
    START_LAYER = None
    NUM_LAYERS = None
    
    # if None, the tree will be collapsed, only takes node embedding in retrieval
    # ablation_name="root_only"
    # COLLAPSE_TREE=False
    # START_LAYER = 3
    # NUM_LAYERS = 1
    
    # ablation_name="leaf_only"
    # COLLAPSE_TREE=False
    # START_LAYER = 0
    # NUM_LAYERS = 1
    
    # ablation_name="hier"
    # COLLAPSE_TREE=False
    # START_LAYER = 3
    # NUM_LAYERS = 4
    
    run_raptor(bench)
import os
import json
from tqdm import tqdm
import argparse
import sys
new_sys_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if new_sys_dir not in sys.path:
    sys.path.append(new_sys_dir)
from raptor import RetrievalAugmentation, RetrievalAugmentationConfig
from raptor.QAModels import BaseQAModel, OpenRouter_QAModel, Local_Qwen2_5_14B_QAModel
from raptor.EmbeddingModels import BaseEmbeddingModel,OpenAIEmbeddingModel,Qwen3Embedding0_6BModel
from datetime import datetime
from f1_em_metrics import f1_score,exact_match_score
import requests

def is_supported(model_id: str) -> bool:
    r = requests.get("https://openrouter.ai/api/v1/models", timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    return any(m.get("id") == model_id for m in data)


def run_raptor(bench,max_examples,qa_model_name,embed_model_name,write_every):
    if bench=="hotpotqa":
        data=json.load(open(EVAL_DATA_PATH,"r"))
        if max_examples>0:
            data=data[:max_examples]
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
    else:
        raise NotImplementedError
    
    if START_LAYER is None and NUM_LAYERS is None:
        res_path=f"{RES_DIR}/{bench}_res.json"
    else:
        res_path=f"{RES_DIR}/{bench}_res_{ablation_name}_start{START_LAYER}_{NUM_LAYERS}layers.json"
    
    QA_model=None
    if qa_model_name in ["qwen2.5-14b-instruct",]:
        QA_model=Local_Qwen2_5_14B_QAModel(model=qa_model_name)
    elif is_supported(qa_model_name):
        QA_model=OpenRouter_QAModel(model=qa_model_name)
    else:
        raise NotImplementedError
    
    embedding_model=None
    if embed_model_name == "text-embedding-3-small":
        embedding_model=OpenAIEmbeddingModel()
    elif embed_model_name in ["qwen/qwen3-embedding-0.6b","qwen3-embedding-0.6b"]:
        embedding_model=Qwen3Embedding0_6BModel(model=embed_model_name)
    else:
        raise NotImplementedError
    
    config = RetrievalAugmentationConfig(qa_model=QA_model,embedding_model=embedding_model)
    RA=None
    if not os.path.exists(TREE_SAVE_PATH):
        RA = RetrievalAugmentation(config=config)
        RA.add_documents(all_context_one_str)
        RA.save(TREE_SAVE_PATH)
    else:
        RA = RetrievalAugmentation(config=config,tree=TREE_SAVE_PATH)
    
    res_list=[]
    gold_answers=[]
    model_answers=[]
    n=0
    template = (
            "You are given retrieved facts from an external memory.\n"
            "Answer the question based on the retrieved facts and your knowledge. \n"
            "Extract substring from the retrieved facts (question not included here) as the answer. If extracting is hard, generate the answer from your own knowledge. \n"
            "And the answer is always a short answer with few words/phrases.\n"
            "DO NOT include anything else like reasoning or process or explanation before or after your answer!! \n\n"
            "Here is the input: \n"
            "Retrieved:\n{information}\n"
            "Question: {question}\n\n"
            "Output format: \n<short answer to the question>\n\n"
        ) 
    for id,q,ref_answer in tqdm(zip(ids,questions,answers)):
        context = RA.retrieve(
            question=q, start_layer=START_LAYER,num_layers=NUM_LAYERS, top_k=10, max_tokens=2048, collapse_tree=COLLAPSE_TREE
        )
        user_prompt=template.format(information=context,question=q)
        model_answer = RA.qa_model.answer_question(user_prompt=user_prompt)

        print("\n")
        print(model_answer)
        print(ref_answer) 
        gold_answers.append(ref_answer)
        model_answers.append(model_answer)           
        res_list.append({
            "id":id,
            "question":q,
            "ref_answer":ref_answer,
            "model_answer":model_answer,
        })
        n += 1
        
        if n % write_every == 0:
            with open(res_path,"w") as f:
                json.dump(res_list,f,indent=4)
            f1=f1_score(gold_answers,model_answers)
            em=exact_match_score(gold_answers,model_answers)
            print(f"[{n}/{len(data)}] EM={em:.4f} F1={f1:.4f}")
            
            
    f1=f1_score(gold_answers,model_answers)
    em=exact_match_score(gold_answers,model_answers)
    print(f"[{n}/{len(data)}] EM={em:.4f} F1={f1:.4f}")

    with open(res_path,"w") as f:
        json.dump(res_list,f,indent=4)
    
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, default="hotpotqa")
    parser.add_argument('--max_examples', type=str, default=100)
    parser.add_argument('--qa_model', type=str, default="gpt-4o-mini")
    parser.add_argument('--embedding_model', type=str, default="text-embedding-3-small")
    parser.add_argument("--write_every", type=int, default=20)
    
    args = parser.parse_args()
    bench=args.bench
    
    if bench=="hotpotqa":
        EVAL_DATA_PATH="data/hotpotqa_1000.json"
    else:
        EVAL_DATA_PATH=None
    max_examples=args.max_examples
    qa_model_name = args.qa_model
    embedding_model_name = args.embedding_model
    write_every = args.write_every
    
    RES_DIR="eval/res"
    TREE_SAVE_PATH = f"eval/tree/{bench}_tree_{embedding_model_name}"
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
    
    run_raptor(bench,max_examples,qa_model_name,embedding_model_name,write_every)
    # python eval/eval_raptor.py --qa_model "qwen2.5-72b-instruct" --embedding_model "qwen/qwen3-embedding-4b"
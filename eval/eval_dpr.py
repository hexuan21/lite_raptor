import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse

from utils import  _llm_openrouter_api,HOTPOTQA_PREFIX,MAX_TRY



@dataclass
class QAItem:
    id: int
    question: str
    answer: str
    context: str


@dataclass
class Chunk:
    chunk_id: int
    source_id: int
    text: str




def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


def build_chunks(items: List[QAItem]) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_id = 0
    for item in items:
        chunk_list = _chunk_text(item.context, CHUNK_SIZE, CHUNK_OVERLAP)

        for c in chunk_list:
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_id=item.id,
                    text=c,
                )
            )
            chunk_id += 1
    print(f"Built {len(chunks)} chunks from {len(items)} contexts.")
    return chunks



class EmbeddingIndex:
    def __init__(self, model_name: str):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunk_texts: List[str] = []
        self.chunk_meta: List[Chunk] = []

    def build(self, chunks: List[Chunk]):
        self.chunk_texts = [c.text for c in chunks]
        self.chunk_meta = chunks

        print("Encoding chunks...")
        embeddings = self.model.encode(
            self.chunk_texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        print(f"Embeddings shape: {embeddings.shape}")

        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query: str, top_k: int) -> List[Tuple[Chunk, float]]:
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        scores, indices = self.index.search(query_emb, top_k)
        scores = scores[0]
        indices = indices[0]

        results: List[Tuple[Chunk, float]] = []
        for idx, s in zip(indices, scores):
            if idx == -1:
                continue
            results.append((self.chunk_meta[idx], float(s)))
        return results




def rag_answer_with_llm(
    question: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
    model_name: str = "gpt-4o-mini",
    max_context_len: int = 2000,
    max_try: int = 3,
) -> str:
    
    context_parts = []
    total_len = 0
    for chunk, score in retrieved_chunks:
        t = chunk.text.replace("\n", " ").strip()
        if not t:
            continue
        if total_len + len(t) > max_context_len:
            break
        context_parts.append(f"[Score {score:.3f}] {t}")
        total_len += len(t)

    context = "\n\n".join(context_parts)

    prefix=""
    if BENCH=="hotpotqa":
        prefix=HOTPOTQA_PREFIX
    else:
        pass
    
    prompt=prefix+f"\n\nContext:{context}.\n\nQuestion: {question}. "

    try_counts=0
    while True:
        if try_counts>=max_try:
            break

        answer = _llm_openrouter_api(prompt,model_name)
        try_counts+=1
        if len(answer)>0:
            break
        
        print(f"trying again, {try_counts}-try (max:{max_try} times)")
    
    return answer



def main():
    if BENCH=="hotpotqa":
        data=json.load(open(EVAL_DATA_PATH,"r"))
        ids=[x['_id'] for x in data]
        questions=[x['question'] for x in data]
        answers=[x['answer'] for x in data]
        all_contexts=[]
        for x in data:
            context_str=""
            for y in x["context"]:
                context_str+="\nCaption:"+y[0]+"\nContent:\n"+"".join(y[1]) 
            all_contexts.append(context_str)
        
    qa_items=[QAItem(id=id, question=q, answer=a, context=ctx,)
        for id,q,a,ctx in zip(ids,questions,answers,all_contexts)
    ]
    chunks = build_chunks(qa_items)
    index = EmbeddingIndex(EMBED_MODEL_NAME)
    index.build(chunks)

    res_list=[]
    for item in tqdm(qa_items):
        q=item.question
        retrieved = index.search(q, top_k=TOP_K)
        model_answer = rag_answer_with_llm(q, retrieved, MODEL_NAME)
        
        res_list.append({
            "id":item.id,
            "question":q,
            "ref_answer":item.answer,
            "model_answer":model_answer,
        })
    
    res_path=f"{RES_DIR}/{BENCH}_res_bsline_dpr.json"
    with open(res_path,"w") as f:
        json.dump(res_list,f,indent=4)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', type=str, default="hotpotqa")
    parser.add_argument('--model_name', type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    BENCH=args.bench
    if BENCH=="hotpotqa":
        EVAL_DATA_PATH="data/hotpotqa_1000.json"
    else:
        EVAL_DATA_PATH=None
    MODEL_NAME=args.model_name
    
    RES_DIR="eval/res"
    EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    TOP_K = 5          
    CHUNK_SIZE = 500    
    CHUNK_OVERLAP = 100  
    
    main()

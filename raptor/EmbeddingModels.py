import logging
from abc import ABC, abstractmethod
import os
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

MAX_TRY=3
import time

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self, text):
        pass


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model="text-embedding-3-small"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url=os.environ["OPENAI_BASE_URL"])
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model)
            .data[0]
            .embedding
        )

class Qwen3Embedding8BModel(BaseEmbeddingModel):
    def __init__(self, model="Qwen/Qwen3-Embedding-8B"):
        self.client = None
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        model_name = "Qwen/Qwen3-Embedding-8B"
        url = "https://hyperdiastolic-nonfavorable-bernard.ngrok-free.dev/v1/embeddings"
        
        for attempt in range(1, MAX_TRY + 1):
            try:
                headers = {"Content-Type": "application/json",}
                data = {"model": model_name,"input": text,}
                response = requests.post(url, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                embedding = result["data"][0]["embedding"]
                print("√ Embedding generated")
                return embedding

            except Exception as e:
                print(f"[Attempt {attempt}/{MAX_TRY}] error: {repr(e)}", flush=True)

            time.sleep(5.0)

        return None
        

class NV_Embed_v2_Model(BaseEmbeddingModel):
    def __init__(self, model="nvidia/NV-Embed-v2"):
        self.client = None
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        model_name = "nvidia/NV-Embed-v2"
        url = "https://tiara-untemperable-inerasably.ngrok-free.dev/v1/embeddings"
        
        for attempt in range(1, MAX_TRY + 1):
            try:
                headers = {"Content-Type": "application/json",}
                data = {"model": model_name,"input": text,}
                response = requests.post(url, json=data, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                embedding = result["data"][0]["embedding"]
                print("√ Embedding generated")
                return embedding

            except Exception as e:
                print(f"[Attempt {attempt}/{MAX_TRY}] error: {repr(e)}", flush=True)

            time.sleep(5.0)

        return None


class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

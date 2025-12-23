import logging
from abc import ABC, abstractmethod
import os
import requests
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


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

class Qwen3Embedding0_6BModel(BaseEmbeddingModel):
    def __init__(self, model="qwen/qwen3-embedding-0.6b"):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"],base_url=os.environ["OPENAI_BASE_URL"])
        self.model = model

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text = text.replace("\n", " ")
        model_name = "Qwen/Qwen3-Embedding-0.6B"
        url = "https://hyperdiastolic-nonfavorable-bernard.ngrok-free.dev/v1/embeddings"
        
        headers = {"Content-Type": "application/json",
                #    "Connection": "close",
                }

        data = {
            "model": model_name,
            "input": text,
        }
        
        try:
            response = requests.post(url, json=data, headers=headers, timeout=60)
            response.raise_for_status()
            result = response.json()

            if "data" in result and result["data"]:
                embedding = result["data"][0]["embedding"]
                print(f"✓ Embedding generated")
                return embedding
            else:
                raise ValueError(f"Bad response format: {result}")

        except requests.exceptions.SSLError as e:
            raise ValueError(f"SSL error: {e}")
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Request error: {e}")
        
        # return (
        #     self.client.embeddings.create(input=[text], model=self.model)
        #     .data[0]
        #     .embedding
        # )

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)

    def create_embedding(self, text):
        return self.model.encode(text)

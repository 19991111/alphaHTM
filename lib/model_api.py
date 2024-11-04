from openai import OpenAI
from transformers import AutoTokenizer
import logging

def load_local_LLM(model_ckpt):
    if model_ckpt == "Qwen/Qwen2.5-14B-Instruct":
        model = "lmstudio-community/Qwen2.5-14B-Instruct-GGUF"
    elif model_ckpt == "mistralai/Mistral-7B-Instruct-v0.3":
        model = "lmstudio-community/Mistral-7B-Instruct-v0.3-GGUF"
    return model

def get_tokenizer(model_ckpt):
    logging.getLogger("transformers").setLevel(logging.ERROR)
    if model_ckpt == "Qwen/Qwen2.5-14B-Instruct":
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt,trust_remote_code=True,add_special_tokens=False)
    elif model_ckpt == "mistralai/Mistral-7B-Instruct-v0.3":
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt,trust_remote_code=True,add_special_tokens=False)
    return tokenizer

class LLM():
    def __init__(self,model_ckpt) -> None:
        self.nomic_text_embedding_model = "nomic-ai/nomic-embed-text-v1.5-GGUF"
        self.model = load_local_LLM(model_ckpt)
        self.client = OpenAI(base_url="http://localhost:1234/v1",api_key="lm_studio")

    def local_LLM_generate(self,temperature,top_p,max_tokens,prompt):
        response = self.client.chat.completions.create(
            model = self.model,
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            temperature = float(temperature),
            max_tokens=int(max_tokens),
            top_p=float(top_p),
        )
        return response.choices[0].message.content

    def get_topics_embedding(self,anytopicslist):
        topic_embedding_list = [self.client.embeddings.create(input = topic, model=self.nomic_text_embedding_model).data[0].embedding for topic in anytopicslist]
        return topic_embedding_list
    
    def get_doc_embedding(self,doc):
        return self.client.embeddings.create(input = doc, model=self.nomic_text_embedding_model).data[0].embedding







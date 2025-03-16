from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.utils import disable_torch_init
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

model_name = "sentence-transformers/all-mpnet-base-v2"  
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name).to(device)


def append_to_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data, ensure_ascii=False) + '\n')  
        

def preprocess_images(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    return images
        

def compute_query_embedding(query, model, device):
    query_embedding = model.encode(
        query, convert_to_tensor=True, normalize_embeddings=True, device=device
    )
    return query_embedding



def compute_candidate_embeddings(candidates, model, device):
    candidate_embeddings = model.encode(
        candidates, convert_to_tensor=True, normalize_embeddings=True, device=device
    )
    return candidate_embeddings



def find_top_k_similar_texts(query, candidate_embeddings, model, device, top_k=5):
    query_embedding = compute_query_embedding(query, model, device)
    similarities = util.cos_sim(query_embedding, candidate_embeddings).squeeze(0)
    top_k_values, top_k_indices = torch.topk(similarities, top_k)
    return top_k_indices
        
  
def get_top_k_RAG_text_only():
    
    eval_data = []
    with open("evoke_evaluation_data.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            eval_data.append(json.loads(line))
    print(len(eval_data))
    
    with open("evoke_injection_data.json", 'r', encoding='utf-8') as file:
        train_data= json.load(file)
    print(len(train_data))
    
    train_texts = []
    for da in train_data:
        train_texts.append(da['conversations'][1]['value'])

    candidate_embeddings = compute_candidate_embeddings(train_texts, model, device)
    
    for da in tqdm(eval_data):
        question = da['text']
        top_k_10 = find_top_k_similar_texts(question, candidate_embeddings, model, device, top_k=10)
        da['top_k_text_only'] = top_k_10.cpu().tolist()
        append_to_jsonl_file(os.path.join("retrieval_data", "retrieval_data_with_top_k_text_only.jsonl"), da)      
        

get_top_k_RAG_text_only()




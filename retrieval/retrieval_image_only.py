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
from tqdm import tqdm


model_name = "openai/clip-vit-base-patch32"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = CLIPModel.from_pretrained(model_name).to(device)  
processor = CLIPProcessor.from_pretrained(model_name)


def append_to_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data, ensure_ascii=False) + '\n')  
        

def preprocess_images(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    return images


def compute_single_image_embedding(image_path, model, processor, device):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device) 
    with torch.no_grad():
        embedding = model.get_image_features(**inputs)  
    return embedding


def compute_batch_image_embeddings(image_paths, model, processor, device):
    images = preprocess_images(image_paths)
    inputs = processor(images=images, return_tensors="pt", padding=True).to(device)  
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)  
    return embeddings


def cosine_similarity(query_embedding, candidate_embeddings):
    query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
    candidate_embeddings = candidate_embeddings / candidate_embeddings.norm(dim=-1, keepdim=True)
    return torch.matmul(query_embedding, candidate_embeddings.T)


def find_most_similar_image(query_image_path, candidate_embeddings, top_k=10):
    query_embedding = compute_single_image_embedding(query_image_path, model, processor, device)
    similarities = cosine_similarity(query_embedding, candidate_embeddings).squeeze(0)
    top_k_values, top_k_indices = torch.topk(similarities, top_k)  
    return top_k_indices



def get_top_k_RAG_image_only():
    
    eval_data = []
    with open("evoke_evaluation_data.jsonl", 'r', encoding='utf-8') as file:
        for line in file:
            eval_data.append(json.loads(line))
    print(len(eval_data))
    
    with open("evoke_injection_data.json", 'r', encoding='utf-8') as file:
        train_data= json.load(file)
    print(len(train_data))
    
    train_imgs = []
    for da in tqdm(train_data):
        cur_img = os.path.join("injection", da['image']) # The image path for injecting knowledge includes:evoke_entity_injection_imgs and evoke_news_injection_imgs
        cur_candidate_embeddings = compute_single_image_embedding(cur_img, model, processor, device)
        train_imgs.append(cur_candidate_embeddings)
    
    candidate_embeddings = torch.cat(train_imgs)
    for da in tqdm(eval_data):
        test_img = os.path.join("evaluation", da['image']) # The image path for injecting knowledge includes:evoke_news_evaluation_imgs and evoke_entity_evaluation_imgs
        top_k_10 = find_most_similar_image(test_img, candidate_embeddings)   
        da['top_k_image_only'] = top_k_10.cpu().tolist()
        append_to_jsonl_file(os.path.join("retrieval_data", "retrieval_data_with_top_k_image_only.jsonl"), da)
        
get_top_k_RAG_image_only()



import argparse
import json
import os
import torch
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from llava.utils import disable_torch_init

def disable_torch():
    disable_torch_init()

def append_to_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data, ensure_ascii=False) + '\n')  

def preprocess_images(image_paths):
    images = [Image.open(path).convert("RGB") for path in image_paths]
    return images

def get_llava_output(test_type, top_k):
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base=None, model_name=model_name
    )

    if test_type == "image_only":
        file_path = "retrieval/retrieval_data/retrieval_data_with_top_k_image_only.jsonl"
    elif test_type == "text_only":
        file_path = "retrieval/retrieval_data/retrieval_data_with_top_k_text_only.jsonl"
    elif test_type == "UniIR":
        file_path = "retrieval/retrieval_data/retrieval_data_with_top_k_UniIR.jsonl"
    elif test_type == "ground_truth":
        file_path = "retrieval/retrieval_data/retrieval_data_with_golden_context.jsonl"
        top_k = 1
    else:
        raise ValueError("Invalid test_type")

    eval_data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            eval_data.append(json.loads(line))
    
    with open("evoke_injection_data.json", 'r', encoding='utf-8') as file:
        train_data= json.load(file)

    index_self = {0: "first", 1: "second", 2: "third", 3: "fourth", 4: "fifth", 5: "sixth"}
    
    for da in eval_data:
        question = da['text']
        test_img = os.path.join("/home/du/pan1/new_knowledge/real_new/all_dataset_self/jkl_data/1_15_new_training_data", da['image'])
        
        if test_type == "image_only":
            can_list = da['top_k_image_only'][0:top_k]
        elif test_type == "text_only":
            can_list = da['top_k_text_only'][0:top_k]
        elif test_type == "UniIR":
            can_list = da['top_k_UniIR'][0:top_k]
        elif test_type == "ground_truth":
            can_list = da['ground_truth'][0:top_k]

        image_file = ""
        prompt = ""
        for i in range(top_k):
            if i == 0:
                image_file += os.path.join("image_path", train_data[can_list[i]]['image'])
            else:
                image_file += ";;" + os.path.join("image_path", train_data[can_list[i]]['image'])
            prompt += "The {} image is about an object of {}. {}".format(
                index_self[i], da['type'], train_data[can_list[i]]['conversations'][1]['value']
            ) + "\n"
        prompt += "\n" + question
        image_file += ";;" + os.path.join("image_path", da['image'])

        args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "model_name": model_name,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "sep": ";;",
            "temperature": 0,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        out = eval_model(args, tokenizer, model, image_processor, context_len, model_name)
        
        new_temp = {
            'question_id': da['question_id'],
            'prompt': da['text'],
            'text': out,
            'type': da['type'],
            'gt_answer': da['answer']
        }
        
        append_to_jsonl_file("res/res_{}_{}.jsonl".format(test_type, top_k), new_temp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLava evaluation with different parameters.")
    parser.add_argument("--test_type", type=str, required=True, choices=["image_only", "text_only", "UniIR", "ground_truth"],
                        help="Type of test to run")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top-k elements to consider")
    args = parser.parse_args()

    get_llava_output(args.test_type, args.top_k)
import json

def append_to_jsonl_file(file_path, new_data):
    with open(file_path, 'a', encoding='utf-8') as file:
        file.write(json.dumps(new_data, ensure_ascii=False) + '\n')  

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))
    return data

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    return data

# # 使用函数读取文件
# file_path = '/home/du/new_models/TIGER-Lab/M-BEIR/query/test/mbeir_oven_task8_test.jsonl'
# data = read_jsonl(file_path)
# print(len(data))

# need_list = []
# for da in data:
#     need_list.append(da['qid'])

# file_path_out = '/home/du/new_models/TIGER-Lab/M-BEIR/query/test/mbeir_oven_task8_test_short.jsonl'
# for index, da in enumerate(data):
#     # del da['query_src_content']
#     # del da['task_id']
#     if index > 20:
#         continue
#     append_to_jsonl_file(file_path_out, da)

# file_path = '/home/du/new_models/TIGER-Lab/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool.jsonl'
# data = read_jsonl(file_path)
# print(len(data))

# need_list_ans = []
# for da in data:
#     if da['did'] == '5:968730':
#         a = 1
#     need_list_ans.append(da['did'])



# file_path_out = '/home/du/new_models/TIGER-Lab/M-BEIR/cand_pool/local/mbeir_oven_task8_cand_pool_short.jsonl'
# for index, da in enumerate(data):
#     # del da['src_content']
#     if index > 20:
#         continue
#     append_to_jsonl_file(file_path_out, da)
    
    
    
#     _task8_test_qrels.txt
# Retriever: Average number of relevant documents per query: 17.72
# Retriever: Searching with k=10
# Faiss: loaded query embeddings from /home/du/pan1/new_knowledge/real_new/mmrag/UniIR/embed/CLIP_SF/Large/Instruct/UniRAG/test/mbeir_oven_task8_test_embed.npy with shape: (14741, 768)
# Faiss: loaded index from /home/du/pan1/new_knowledge/real_new/mmrag/UniIR/index/CLIP_SF/Large/Instruct/UniRAG/cand_pool/mbeir_union_cand_pool.index
# Faiss: Number of documents in the index: 335135
# Faiss: Number of GPUs used for searching: 1
# Faiss: query_embeddings_batch.shape: (14741, 768)
# Retriever: Run file saved to /home/du/pan1/new_knowledge/real_new/mmrag/UniIR/retrieval_results/CLIP_SF/Large/Instruct/UniRAG/run_files/mbeir_oven_task8_union_pool_test_k10_run.txt
# Retriever: Mean Recall@1: 0.4935
# Retriever: Mean Recall@5: 0.6874
# Retriever: Mean Recall@10: 0.7441

import os
task_name = "new_self"
## query


file_path = "/home/du/pan1/new_knowledge/real_new/all_dataset_self/jkl_data/version_1_data/eval_vqa.jsonl"
data = read_jsonl(file_path)
print(len(data))
file_path_out = "/home/du/new_models/TIGER-Lab/M-BEIR/query/test/mbeir_{}_test.jsonl".format(task_name)
if os.path.exists(file_path_out):
    os.remove(file_path_out)
q_id = 5 # or 6
for da in data:
    temp_dict = {}
    temp_dict['qid'] = "{}:{}".format(q_id, da['question_id'])
    temp_dict['query_txt'] = da['text']
    temp_dict['query_img_path'] = da['image']
    temp_dict['query_modality'] = "image,text"
    temp_dict["pos_cand_list"] = ["{}:{}".format(q_id, da['question_id'])]
    temp_dict["neg_cand_list"] = []
    append_to_jsonl_file(file_path_out, temp_dict)

### cand
file_path = "/home/du/pan1/new_knowledge/real_new/all_dataset_self/jkl_data/version_1_data/random_img_training_data.jsonl"
data = read_json(file_path)
print(len(data))
file_path_out = "/home/du/new_models/TIGER-Lab/M-BEIR/cand_pool/local/mbeir_{}_cand_pool.jsonl".format(task_name)
if os.path.exists(file_path_out):
    os.remove(file_path_out)
q_id = 5
for da in data:
    temp_dict = {}
    temp_dict['txt'] = da['conversations'][1]['value']
    temp_dict['img_path'] = da['image']
    temp_dict["modality"] = "image,text"
    temp_dict["did"] = "{}:{}".format(q_id, da['id'])
    append_to_jsonl_file(file_path_out, temp_dict)
    
### qrel
file_path = "/home/du/new_models/TIGER-Lab/M-BEIR/query/test/mbeir_new_self_test.jsonl"
data = read_jsonl(file_path)
print(len(data))
with open("/home/du/new_models/TIGER-Lab/M-BEIR/qrels/test/mbeir_{}_test_qrels.txt".format(task_name), 'w') as f:
    for da in data:
        for tt in da['pos_cand_list']:
            f.write("{} 0 {} 1 26".format(da['qid'], tt))
            f.write("\n")
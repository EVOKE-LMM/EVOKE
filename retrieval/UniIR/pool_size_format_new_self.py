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


import os
task_name = "new_self"


file_path = "/home/du/pan1/new_knowledge/real_new/all_dataset_self/jkl_data/version_1_data/used_for_pool_size/new_eval_0_2.jsonl"
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
file_path = "/home/du/pan1/new_knowledge/real_new/all_dataset_self/jkl_data/version_1_data/used_for_pool_size/new_train_0_10.jsonl"
data = read_jsonl(file_path)
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
import json
from collections import defaultdict

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_to_txt(file_path, content):
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)

def process_files(file_a_path, file_b_path, output_txt_path):

    data_a = read_jsonl(file_a_path)
    data_b = read_jsonl(file_b_path)

    b_data_dict = {item['question_id']: item for item in data_b}

    wiki_data = []
    cnn_data = []

    for item in data_a:
        question_id = item['question_id']
        if question_id in b_data_dict:
            merged_data = {**item, **b_data_dict[question_id]}
            if item['cnn_wiki_type'] == 'wiki':
                wiki_data.append(merged_data)
            elif item['cnn_wiki_type'] == 'news':
                cnn_data.append(merged_data)

    def calculate_average(data_list):
        if not data_list:
            return 0, 0
        total_score = sum(item['score'] for item in data_list)
        total_f1_score = sum(item['f1_score'] for item in data_list)
        return total_score / len(data_list), total_f1_score / len(data_list)

    wiki_avg_score, wiki_avg_f1_score = calculate_average(wiki_data)
    cnn_avg_score, cnn_avg_f1_score = calculate_average(cnn_data)

    def calculate_type_statistics(data_list):
        type_stats = defaultdict(list)
        for item in data_list:
            type_stats[item['type']].append(item)
        stats_result = []
        type_count = len(type_stats)
        for type_key, items in sorted(type_stats.items(), key=lambda x: len(x[1]), reverse=True):
            avg_score, avg_f1_score = calculate_average(items)
            stats_result.append(f"{type_key} {len(items)} {avg_score:.4f} {avg_f1_score:.4f}")
        print(f"所有类型的总数: {type_count}")
        return "\n".join(stats_result)

    wiki_type_stats = calculate_type_statistics(wiki_data)
    cnn_type_stats = calculate_type_statistics(cnn_data)

    output_content = (
        f"Wiki平均分:\nScore: {wiki_avg_score:.4f}, F1_Score: {wiki_avg_f1_score:.4f}\n\n"
        f"CNN平均分:\nScore: {cnn_avg_score:.4f}, F1_Score: {cnn_avg_f1_score:.4f}\n\n"
        f"Wiki类型统计:\n{wiki_type_stats}\n\n"
        f"CNN类型统计:\n{cnn_type_stats}\n"
    )
    write_to_txt(output_txt_path, output_content)
    print(f"结果已写入 {output_txt_path}")


file_a_path = "evoke_evaluation_data.jsonl"

file_b_path = "merge_eval_acc_f1.jsonl"
output_file_name = "type_score_output.txt"

import os

base_name = os.path.basename(file_b_path)

output_file_name = base_name.split('.')[0] + "_" + output_file_name

output_txt_path = os.path.join(os.path.dirname(file_b_path), output_file_name)

process_files(file_a_path, file_b_path, output_txt_path)

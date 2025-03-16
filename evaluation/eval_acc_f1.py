import json
from tools import VQAEval
import re
import string
import collections
import os
eval_tool = VQAEval()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def get_f1_score(a_pred, a_gold):
    """Calculate F1 score between predicted and gold answers."""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def evaluate_vqa(a_file, b_file, output_file):
    total_score = 0
    total_f1_score = 0
    num_entries = 0

    with open(a_file, 'r', encoding='utf-8') as af:
        a_entries = [json.loads(line) for line in af]
        a_data = {entry['question_id']: entry['answer'] for entry in a_entries}
        a_data_type = {entry['question_id']: entry['type'] for entry in a_entries}

    updated_b_data = []

    with open(b_file, 'r', encoding='utf-8') as bf:
        for line in bf:
            b_entry = json.loads(line)
            qid = b_entry['question_id']
            pred_answer = b_entry['text']
            # pred_answer = b_entry['answer']
            gt_answer = a_data.get(qid, "")  
            type = a_data_type.get(qid, "")  

            score = eval_tool.evaluate(pred_answer, [gt_answer])
            f1_score = get_f1_score(pred_answer, gt_answer)
            
            b_entry['score'] = score
            b_entry['f1_score'] = f1_score  # Add F1 score
            b_entry['gt_answer'] = gt_answer  
            b_entry['type'] = type
            
            total_score += score
            total_f1_score += f1_score
            num_entries += 1

            updated_b_data.append(b_entry)
            


    avg_score = total_score / num_entries if num_entries > 0 else 0
    avg_f1_score = total_f1_score / num_entries if num_entries > 0 else 0

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for entry in updated_b_data:
            out_f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"\nTotal Score: {total_score}, Total F1-Score: {total_f1_score:.4f}")
    print(f"Average Score: {avg_score:.4f}, Average F1-Score: {avg_f1_score:.4f}")
    return avg_score, avg_f1_score


a_file_path = "evoke_evaluation_data.jsonl"  


b_file_path = "merge.jsonl"
output_file_name = "merge_eval_acc_f1.jsonl"  

output_file_path = os.path.join(os.path.dirname(b_file_path), output_file_name)

evaluate_vqa(a_file_path, b_file_path, output_file_path)

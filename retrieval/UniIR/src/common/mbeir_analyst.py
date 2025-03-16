import os
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from datetime import datetime

import numpy as np
import csv

from data.preprocessing.utils import (
    load_jsonl_as_list,
    save_list_as_jsonl,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_dataset_stats,
    unhash_did,
    unhash_qid,
    get_dataset_name,
    MBEIR_DATASET_TO_DOMAIN,
    get_mbeir_task_name,
    get_mbeir_query_modality_cand_modality_from_task_id,
)
from utils import (
    load_qrel,
    load_runfile,
)


def run_automatic_error_analysis(config):
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    expt_dir_name = config.experiment.path_suffix
    analysis_config = config.analysis_config
    # cand_pool_dir = config.data_config.cand_pool_dir_name
    qrel_dir_name = analysis_config.qrel_dir_name
    results_dir_name = analysis_config.results_dir_name
    exp_results_dir = os.path.join(uniir_dir, results_dir_name, expt_dir_name)
    exp_run_file_dir = os.path.join(exp_results_dir, "run_files")
    exp_error_tsv_results_dir = os.path.join(exp_results_dir, "error_tsv")
    os.makedirs(exp_error_tsv_results_dir, exist_ok=True)

    splits = []
    # Load the dataset splits to analysis
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        analysis_dataset_config = getattr(analysis_config, f"{split_name}_datasets_config", None)
        if analysis_dataset_config and analysis_dataset_config.enable_retrieve:
            dataset_name_list = getattr(analysis_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(analysis_dataset_config, "correspond_cand_pools_name", None)
            qrel_name_list = getattr(analysis_dataset_config, "correspond_qrels_name", None)
            metric_names_list = getattr(analysis_dataset_config, "correspond_metrics_name", None)
            splits.append(
                (
                    split_name,
                    dataset_name_list,
                    cand_pool_name_list,
                    qrel_name_list,
                    metric_names_list,
                )
            )
            assert (
                len(dataset_name_list) == len(cand_pool_name_list) == len(qrel_name_list) == len(metric_names_list)
            ), "Mismatch between datasets and candidate pools and qrels."

    # Pretty Print dataset to index
    print("-" * 30)
    for (
        split_name,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        print(
            f"Split: {split_name}, Retrieval Datasets: {dataset_name_list}, Candidate Pools: {cand_pool_name_list}, Metric: {metric_names_list})"
        )
        print("-" * 30)

    eval_results = []
    union_pool_cache = None
    qrel_dir = os.path.join(mbeir_data_dir, qrel_dir_name)
    for (
        split,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        for dataset_name, cand_pool_name, qrel_name, metric_names in zip(
            dataset_name_list, cand_pool_name_list, qrel_name_list, metric_names_list
        ):
            print("\n" + "-" * 30)
            print(f"Error Analyst: Analysis for query:{dataset_name} | split:{split} | from cand_pool:{cand_pool_name}")

            dataset_name = dataset_name.lower()
            cand_pool_name = cand_pool_name.lower()
            qrel_name = qrel_name.lower()

            # Load qrels
            qrel_path = os.path.join(qrel_dir, f"mbeir_{qrel_name}_{split}_qrels.txt")
            qrel, qid_to_taskid = load_qrel(qrel_path)

            # Load the metric
            metric_list = [metric.strip() for metric in metric_names.split(",")]
            metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]
            k = max([int(metric.split("@")[1]) for metric in metric_recall_list])

            # Open a file to write the run results
            if cand_pool_name == "union":
                run_id = f"mbeir_{dataset_name}_union_pool_{split}_k{k}"
            else:
                run_id = f"mbeir_{dataset_name}_single_pool_{split}_k{k}"
            run_file_name = f"{run_id}_run.txt"
            run_file_path = os.path.join(exp_run_file_dir, run_file_name)

            # Load the run file
            run_results = load_runfile(run_file_path, load_task_id=True)
            print(f"Load {len(run_results)} run results from {run_file_path}")

            # Compute errors
            # Load query file and candidate pool file
            split_dir = os.path.join(mbeir_data_dir, split)
            dataset_name = dataset_name.lower()
            query_data_name = f"mbeir_{dataset_name}_{split}.jsonl"
            query_data_path = os.path.join(split_dir, query_data_name)
            query_data = load_jsonl_as_list(query_data_path)

            # Construct the candidate pool path
            cand_pool_name = cand_pool_name.lower()
            if cand_pool_name == "union":
                cand_pool_dir = os.path.join(mbeir_data_dir, "cand_pool", "union_pool")
                cand_pool_file_name = f"mbeir_union_test_cand_pool.jsonl"
                if union_pool_cache is None:
                    cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                        os.path.join(cand_pool_dir, cand_pool_file_name),
                        doc_key_to_content=True,
                        key_type="did",
                    )
                    union_pool_cache = cand_pool_dict
                else:
                    cand_pool_dict = union_pool_cache
            else:
                cand_pool_dir = os.path.join(mbeir_data_dir, "cand_pool")
                cand_pool_file_name = f"mbeir_{cand_pool_name}_cand_pool.jsonl"
                cand_pool_data_path = os.path.join(cand_pool_dir, cand_pool_file_name)
                cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                    cand_pool_data_path, doc_key_to_content=True, key_type="did"
                )

            error_values_by_task = defaultdict(lambda: defaultdict(list))
            error_list = ["Type1", "Type2", "Type3"]
            total_num_false_positives = 0
            for query_entry in query_data:
                qid = query_entry["qid"]
                query_modality, gt_candidate_modality = get_mbeir_query_modality_cand_modality_from_task_id(
                    int(qid_to_taskid[qid])
                )
                assert (
                    query_modality == query_entry["query_modality"]
                ), "Mismatch between query modality in query file and task id."
                task_id = qid_to_taskid[qid]
                error_values_for_qid = {
                    "Type1": 0,
                    "Type2": 0,
                    "Type3": 0,
                }
                num_false_positives = 0
                for run_result in run_results[qid]:
                    if run_result["rank"] == 1:
                        did = run_result["did"]
                        cand = cand_pool_dict[did]
                        cand_modality = cand["modality"]
                        if did in query_entry["pos_cand_list"]:
                            pass  # Correct
                        else:  # Error
                            num_false_positives += 1
                            if gt_candidate_modality != cand_modality:  # Retrieve wrong modality
                                error_values_for_qid["Type1"] += 1
                            elif (
                                MBEIR_DATASET_TO_DOMAIN[get_dataset_name(qid)]
                                != MBEIR_DATASET_TO_DOMAIN[get_dataset_name(did)]
                            ):  # Retrieve wrong domain
                                error_values_for_qid["Type2"] += 1
                            else:
                                error_values_for_qid["Type3"] += 1
                        # Only consider the top ranked result
                        break
                total_num_false_positives += num_false_positives
                for error_type in error_list:
                    error_values_by_task[task_id][error_type].append(error_values_for_qid[error_type])

            print(
                "Error Analyst: Total number of false positives: ",
                total_num_false_positives,
            )

            for task_id, errors in error_values_by_task.items():
                task_name = get_mbeir_task_name(int(task_id))
                result = {
                    "TaskID": int(task_id),
                    "Task": task_name,
                    "Dataset": dataset_name,
                    "Split": split,
                    "CandPool": cand_pool_name,
                }
                for error_type in error_list:
                    mean_error = round(sum(errors[error_type]) / total_num_false_positives, 4)
                    result[error_type] = mean_error
                    print(f"Error Analyst: Mean {error_type}: {mean_error}")
                # assert result["Type1"] + result["Type2"] + result["Type3"] == 1.0, "Error values do not sum to 1.0"
                eval_results.append(result)

    # Creating a tsv file for the evaluation results
    # Sort data by TaskID, then by dataset_order, then by split_order matching the google sheets.
    dataset_order = {
        "visualnews_task0": 1,
        "mscoco_task0": 2,
        "fashion200k_task0": 3,
        "webqa_task1": 4,
        "edis_task2": 5,
        "webqa_task2": 6,
        "visualnews_task3": 7,
        "mscoco_task3": 8,
        "fashion200k_task3": 9,
        "nights_task4": 10,
        "oven_task6": 11,
        "infoseek_task6": 12,
        "fashioniq_task7": 13,
        "cirr_task7": 14,
        "oven_task8": 15,
        "infoseek_task8": 16,
    }
    split_order = {"val": 1, "test": 2}
    cand_pool_order = {"union": 99}
    eval_results_sorted = sorted(
        eval_results,
        key=lambda x: (
            x["TaskID"],
            dataset_order.get(x["Dataset"].lower(), 99),  # default to 99 for datasets not listed
            split_order.get(x["Split"].lower(), 99),  # default to 99 for splits not listed
            cand_pool_order.get(x["CandPool"].lower(), 0),
        ),
    )

    grouped_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    # Populate the grouped_results with metric values
    available_recall_metrics = ["Type1", "Type2", "Type3"]
    for result in eval_results_sorted:
        key = (result["TaskID"], result["Task"], result["Dataset"], result["Split"])
        for metric in available_recall_metrics:
            grouped_results[key][result["CandPool"]].update({metric: result.get(metric, None)})

    # Write the sorted data to TSV
    if analysis_config.write_to_tsv:
        # TODO: create a better file name
        date_time = datetime.now().strftime("%m-%d-%H")
        tsv_file_name = f"error_analysis_results_{date_time}.tsv"
        tsv_file_path = os.path.join(exp_error_tsv_results_dir, tsv_file_name)
        tsv_data = []
        header = [
            "TaskID",
            "Task",
            "Dataset",
            "Split",
            "Metric",
            "CandPool",
            "Value",
            "UnionPool",
            "UnionValue",
        ]
        tsv_data.append(header)

        for (task_id, task, dataset, split), cand_pools in grouped_results.items():
            # Extract union results if available
            union_results = cand_pools.get("union", {})
            for metric in available_recall_metrics:
                for cand_pool, metrics in cand_pools.items():
                    if cand_pool != "union":  # Exclude union pool as it's handled separately
                        row = [
                            task_id,
                            task,
                            dataset,
                            split,
                            metric,
                            cand_pool,
                            metrics.get(metric, None),
                        ]
                        # Skip metric if it's not available
                        if row[-1] is None:
                            continue
                        # Add corresponding union metric value if it exists
                        if union_results:
                            row.extend(["union", union_results.get(metric, "N/A")])
                        else:
                            row.extend(["", ""])  # Fill with empty values if no union result
                        tsv_data.append(row)

        # Write to TSV
        with open(tsv_file_path, "w", newline="") as tsvfile:
            writer = csv.writer(tsvfile, delimiter="\t")
            for row in tsv_data:
                writer.writerow(row)

        print(f"Retriever: Results saved to {tsv_file_path}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Pipeline")
    parser.add_argument("--uniir_dir", type=str, default="/data/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="/data/UniIR/mbeir_data")
    parser.add_argument("--config_path", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--run_automatic_error_analysis",
        action="store_true",
        help="Run automatic error analysis.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    print(OmegaConf.to_yaml(config, sort_keys=False))

    if args.run_automatic_error_analysis:
        run_automatic_error_analysis(config)


if __name__ == "__main__":
    main()

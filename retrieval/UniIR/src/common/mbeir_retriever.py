"""
This module contains the code for indexing and retrieval using FAISS based on the MBEIR embeddings.
"""

import os
import argparse
from omegaconf import OmegaConf
from collections import defaultdict
from datetime import datetime
import json

import numpy as np
import csv
import gc

import faiss
import pickle
import torch
import os
import sys
sys.path.append("/home/du/pan1/new_knowledge/real_new/mmrag/UniIR/src")
from data.preprocessing.utils import (
    load_jsonl_as_list,
    save_list_as_jsonl,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_dataset_stats,
    unhash_did,
    unhash_qid,
    get_mbeir_task_name,
)
import dist_utils
from interactive_retriever import InteractiveRetriever


def create_index(config):
    """This script builds the faiss index for the embeddings generated"""
    uniir_dir = config.uniir_dir
    index_config = config.index_config
    embed_dir_name = index_config.embed_dir_name
    index_dir_name = index_config.index_dir_name
    expt_dir_name = config.experiment.path_suffix

    idx_cand_pools_config = index_config.cand_pools_config
    assert idx_cand_pools_config.enable_idx, "Indexing is not enabled for candidate pool"
    split_name = "cand_pool"
    cand_pool_name_list = idx_cand_pools_config.cand_pools_name_to_idx

    # Pretty Print dataset to index
    print("-" * 30)
    print(f"Split: {split_name}, Candidate pool to index: {cand_pool_name_list}")
    print("-" * 30)

    for cand_pool_name in cand_pool_name_list:
        cand_pool_name = cand_pool_name.lower()

        embed_data_file = f"mbeir_{cand_pool_name}_{split_name}_embed.npy"
        embed_data_path = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name, embed_data_file)
        embed_data_hashed_id_file = f"mbeir_{cand_pool_name}_{split_name}_ids.npy"
        embed_data_hashed_id_path = os.path.join(
            uniir_dir,
            embed_dir_name,
            expt_dir_name,
            split_name,
            embed_data_hashed_id_file,
        )

        print(f"Building index for {embed_data_path} and {embed_data_hashed_id_path}")

        # Load the embeddings and IDs from the .npy files
        embedding_list = np.load(embed_data_path).astype("float32")
        hashed_id_list = np.load(embed_data_hashed_id_path)

        # Check unique ids
        assert len(hashed_id_list) == len(set(hashed_id_list)), "IDs should be unique"

        # Normalize the embeddings
        faiss.normalize_L2(embedding_list)

        # Dimension of the embeddings
        d = embedding_list.shape[1]

        # Create the FAISS index on the CPU
        faiss_config = index_config.faiss_config
        assert faiss_config.dim == d, "The dimension of the index does not match the dimension of the embeddings!"
        metric = getattr(faiss, faiss_config.metric)
        cpu_index = faiss.index_factory(
            faiss_config.dim,
            f"IDMap,{faiss_config.idx_type}",
            metric,
        )
        print("Creating FAISS index with the following parameters:")
        print(f"Index type: {faiss_config.idx_type}")
        print(f"Metric: {faiss_config.metric}")
        print(f"Dimension: {faiss_config.dim}")

        # Distribute the index across multiple GPUs
        ngpus = faiss.get_num_gpus()
        print(f"Number of GPUs used for indexing: {ngpus}")
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        index_gpu = faiss.index_cpu_to_all_gpus(cpu_index, co=co, ngpu=ngpus)

        # Add data to the GPU index
        index_gpu.add_with_ids(embedding_list, hashed_id_list)

        # Transfer the GPU index back to the CPU for saving
        index_cpu = faiss.index_gpu_to_cpu(index_gpu)

        # Save the CPU index to disk
        index_path = os.path.join(
            uniir_dir,
            index_dir_name,
            expt_dir_name,
            split_name,
            f"mbeir_{cand_pool_name}_{split_name}.index",
        )
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(index_cpu, index_path)
        print(f"Successfully indexed {index_cpu.ntotal} documents")
        print(f"Index saved to: {index_path}")

        # 1. Delete large objects
        del embedding_list
        del hashed_id_list
        del cpu_index
        del index_gpu
        del index_cpu

        # 2. Force garbage collection
        gc.collect()


# def compute_recall_at_k(relevant_docs, retrieved_indices, k):
#     if not relevant_docs:
#         return 0.0  # Return 0 if there are no relevant documents
#
#     top_k_retrieved_indices_set = set(retrieved_indices[:k])
#     relevant_docs_set = set(relevant_docs)
#
#     assert len(relevant_docs_set) == len(relevant_docs), "Relevant docs should not contain duplicates"
#     assert len(top_k_retrieved_indices_set) == len(
#         retrieved_indices[:k]
#     ), "Retrieved docs should not contain duplicates"
#
#     relevant_retrieved = relevant_docs_set.intersection(top_k_retrieved_indices_set)
#     recall_at_k = len(relevant_retrieved) / len(relevant_docs)
#     return recall_at_k


def compute_recall_at_k(relevant_docs, retrieved_indices, k):
    # Recall used by CLIP and BLIP codebase
    # Return 0 if there are no relevant documents
    if not relevant_docs:
        return 0.0

    # Get the set of indices for the top k retrieved documents
    top_k_retrieved_indices_set = set(retrieved_indices[:k])

    # Convert the relevant documents to a set
    relevant_docs_set = set(relevant_docs)

    # Check if there is an intersection between relevant docs and top k retrieved docs
    # If there is, we return 1, indicating successful retrieval; otherwise, we return 0
    if relevant_docs_set.intersection(top_k_retrieved_indices_set):
        return 1.0
    else:
        return 0.0


def load_qrel(filename):
    qrel = {}
    qid_to_taskid = {}
    with open(filename, "r") as f:
        for line in f:
            query_id, _, doc_id, relevance_score, task_id = line.strip().split()
            if int(relevance_score) > 0:  # Assuming only positive relevance scores indicate relevant documents
                if query_id not in qrel:
                    qrel[query_id] = []
                qrel[query_id].append(doc_id)
                if query_id not in qid_to_taskid:
                    qid_to_taskid[query_id] = task_id
    print(f"Retriever: Loaded {len(qrel)} queries from {filename}")
    print(
        f"Retriever: Average number of relevant documents per query: {sum(len(v) for v in qrel.values()) / len(qrel):.2f}"
    )
    return qrel, qid_to_taskid


def search_index(query_embed_path, cand_index_path, batch_size=10, num_cand_to_retrieve=10):
    # Load the full query embeddings
    query_embeddings = np.load(query_embed_path).astype("float32") ## (4170, 768)
    print(f"Faiss: loaded query embeddings from {query_embed_path} with shape: {query_embeddings.shape}")

    # Normalize the full query embeddings
    faiss.normalize_L2(query_embeddings)

    # Load the saved CPU index from disk
    index_cpu = faiss.read_index(cand_index_path)
    print(f"Faiss: loaded index from {cand_index_path}")
    print(f"Faiss: Number of documents in the index: {index_cpu.ntotal}")

    # Convert the CPU index to multiple GPU indices
    ngpus = faiss.get_num_gpus()
    print(f"Faiss: Number of GPUs used for searching: {ngpus}")
    co = faiss.GpuMultipleClonerOptions()
    co.shard = True  # Use shard to divide the data across the GPUs
    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu, co=co, ngpu=ngpus)  # This shards the index across all GPUs

    all_distances = []
    all_indices = []

    # Process in batches
    for i in range(0, len(query_embeddings), batch_size):
        batch = query_embeddings[i : i + batch_size]
        distances, indices = search_index_with_batch(batch, index_gpu, num_cand_to_retrieve)
        all_distances.append(distances)
        all_indices.append(indices)

    # Stack results for distances and indices
    final_distances = np.vstack(all_distances)
    final_indices = np.vstack(all_indices)

    return final_distances, final_indices


def search_index_with_batch(query_embeddings_batch, index_gpu, num_cand_to_retrieve=10):
    # Ensure query_embeddings_batch is numpy array with dtype float32
    assert isinstance(query_embeddings_batch, np.ndarray) and query_embeddings_batch.dtype == np.float32
    print(f"Faiss: query_embeddings_batch.shape: {query_embeddings_batch.shape}")

    # Query the multi-GPU index
    distances, indices = index_gpu.search(query_embeddings_batch, num_cand_to_retrieve)  # (number_of_queries, k)
    return distances, indices


def get_raw_retrieved_candidates(
    queries_path, candidates_path, retrieved_indices, hashed_query_ids, complement_retriever
):
    # Load raw queries
    qid_to_queries = {}
    with open(queries_path, "r") as f:
        for l in f:
            q = json.loads(l.strip())
            assert q["qid"] not in qid_to_queries, "qids must be unique"
            qid_to_queries[q["qid"]] = q

    # Load raw candidates
    did_to_candidates = {}
    with open(candidates_path, "r") as f:
        for l in f:
            c = json.loads(l.strip())
            assert c["did"] not in did_to_candidates, "dids must be unique"
            did_to_candidates[c["did"]] = c

    retrieved_dict = {}
    complement_queries_list = []  # Used to map complement queries to original qids.
    for idx, indices in enumerate(retrieved_indices):
        retrieved_cands = []
        qid = unhash_qid(hashed_query_ids[idx])
        query = qid_to_queries[qid]
        for hashed_doc_id in indices:
            doc_id = unhash_did(hashed_doc_id)
            retrieved_cands.append(did_to_candidates[doc_id])
        retrieved_dict[qid] = {"query": query, "candidates": retrieved_cands}
        # For each candidate with image/text modality create a complement query to retrieve the candidate's complement candidate with text/image modality.
        if complement_retriever:
            complement_modalities = {"text": "image", "image": "text"}
            complement_queries = [
                (
                    cand.get("modality"),
                    cand.get("txt"),
                    cand.get("img_path"),
                    complement_modalities[cand.get("modality")],
                )
                for cand in retrieved_cands
                if cand["modality"] in complement_modalities.keys()
            ]
            complement_queries_list.append((qid, complement_queries))
            complement_retriever.add_queries(complement_queries)

    # Retrieve complement candidates for all queries at once.
    if complement_retriever:
        retrieved_complements = complement_retriever.retrieve(k=10)
        complement_queries_start_index = 0
        for qid, complement_queries in complement_queries_list:
            complement_candidates = []
            complement_queries_end_index = complement_queries_start_index + len(complement_queries)
            retrieved_comp_cands = retrieved_complements[complement_queries_start_index:complement_queries_end_index]
            complement_queries_start_index = complement_queries_end_index
            for idx, complement_query in enumerate(complement_queries):
                complement_cand = None
                q_modality = complement_query[0]
                for cand in retrieved_comp_cands[idx]:
                    if cand["modality"] == complement_modalities[q_modality]:
                        # The retrieved complement candidate should not be the same as the original query.
                        if (
                            cand.get("img_path")
                            and cand.get("img_path") != retrieved_dict[qid]["query"]["query_img_path"]
                        ):
                            complement_cand = cand
                            break
                        if cand.get("txt") and cand.get("txt") != retrieved_dict[qid]["query"]["query_txt"]:
                            complement_cand = cand
                            break
                if not complement_cand:
                    print(f"retrieved_dict[qid]: {retrieved_dict[qid].__repr__()}")
                    print(f"retrieved_comp_cands: {retrieved_comp_cands[idx]}")
                complement_candidates.append(complement_cand)
            retrieved_dict[qid]["complement_candidates"] = complement_candidates
    return retrieved_dict


def run_retrieval(config, query_embedder_config=None):
    """This script runs retrieval on the faiss index"""
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    qrel_dir_name = retrieval_config.qrel_dir_name
    embed_dir_name = retrieval_config.embed_dir_name
    index_dir_name = retrieval_config.index_dir_name
    query_dir_name = retrieval_config.query_dir_name
    candidate_dir_name = retrieval_config.candidate_dir_name
    expt_dir_name = config.experiment.path_suffix

    # Create results directory if it doesn't exist
    results_dir_name = retrieval_config.results_dir_name
    exp_results_dir = os.path.join(uniir_dir, results_dir_name, expt_dir_name)
    os.makedirs(exp_results_dir, exist_ok=True)
    exp_run_file_dir = os.path.join(exp_results_dir, "run_files")
    os.makedirs(exp_run_file_dir, exist_ok=True)
    exp_retrieved_cands_dir = os.path.join(exp_results_dir, "retrieved_candidates")
    os.makedirs(exp_retrieved_cands_dir, exist_ok=True)
    exp_tsv_results_dir = os.path.join(exp_results_dir, "final_tsv")
    os.makedirs(exp_tsv_results_dir, exist_ok=True)

    splits = []
    # Load the dataset splits to embed
    dataset_types = ["train", "val", "test"]
    for split_name in dataset_types:
        retrieval_dataset_config = getattr(retrieval_config, f"{split_name}_datasets_config", None)
        if retrieval_dataset_config and retrieval_dataset_config.enable_retrieve:
            dataset_name_list = getattr(retrieval_dataset_config, "datasets_name", None)
            cand_pool_name_list = getattr(retrieval_dataset_config, "correspond_cand_pools_name", None)
            qrel_name_list = getattr(retrieval_dataset_config, "correspond_qrels_name", None)
            metric_names_list = getattr(retrieval_dataset_config, "correspond_metrics_name", None)
            dataset_embed_dir = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, split_name)
            splits.append(
                (
                    split_name,
                    dataset_embed_dir,
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
        dataset_embed_dir,
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
    cand_index_dir = os.path.join(uniir_dir, index_dir_name, expt_dir_name, "cand_pool")
    qrel_dir = os.path.join(mbeir_data_dir, qrel_dir_name)
    for (
        split,
        dataset_embed_dir,
        dataset_name_list,
        cand_pool_name_list,
        qrel_name_list,
        metric_names_list,
    ) in splits:
        for dataset_name, cand_pool_name, qrel_name, metric_names in zip(
            dataset_name_list, cand_pool_name_list, qrel_name_list, metric_names_list
        ):
            print("\n" + "-" * 30)
            print(f"Retriever: Retrieving for query:{dataset_name} | split:{split} | from cand_pool:{cand_pool_name}")

            dataset_name = dataset_name.lower()
            cand_pool_name = cand_pool_name.lower()
            qrel_name = qrel_name.lower()

            # Load qrels
            qrel_path = os.path.join(qrel_dir, split, f"mbeir_{qrel_name}_{split}_qrels.txt")  # ????
            qrel, qid_to_taskid = load_qrel(qrel_path)

            # Load query Hashed IDs
            embed_query_id_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{split}_ids.npy")
            hashed_query_ids = np.load(embed_query_id_path)

            # Load query embeddings
            embed_query_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{split}_embed.npy")

            # Load the candidate pool index
            cand_index_path = os.path.join(cand_index_dir, f"mbeir_{cand_pool_name}_cand_pool.index")

            # Set up the metric
            # e.g. "Recall@1, Recall@5, Recall@10"
            # TODO: add more metrics
            metric_list = [metric.strip() for metric in metric_names.split(",")]
            metric_recall_list = [metric for metric in metric_list if "recall" in metric.lower()]

            # Search the index
            k = max([int(metric.split("@")[1]) for metric in metric_recall_list])
            print(f"Retriever: Searching with k={k}")
            retrieved_cand_dist, retrieved_indices = search_index(
                embed_query_path,
                cand_index_path,
                batch_size=hashed_query_ids.shape[0],
                num_cand_to_retrieve=k,
            )  # Shape: (number_of_queries, k)
            import pickle
            with open("data_self.pkl", "wb") as f:
                pickle.dump(retrieved_indices, f)
            # Open a file to write the run results
            if cand_pool_name == "union":
                run_id = f"mbeir_{dataset_name}_union_pool_{split}_k{k}"
            else:
                run_id = f"mbeir_{dataset_name}_single_pool_{split}_k{k}"
            run_file_name = f"{run_id}_run.txt"
            run_file_path = os.path.join(exp_run_file_dir, run_file_name)
            with open(run_file_path, "w") as run_file:
                for idx, (distances, indices) in enumerate(zip(retrieved_cand_dist, retrieved_indices)):
                    qid = unhash_qid(hashed_query_ids[idx])
                    task_id = qid_to_taskid[qid]
                    for rank, (hashed_doc_id, score) in enumerate(zip(indices, distances), start=1):
                        # Format: query-id Q0 document-id rank score run-id task_id
                        # We can remove task_id if we don't need it later using a helper
                        # Note: since we are using the cosine similarity, we don't need to invert the scores.
                        doc_id = unhash_did(hashed_doc_id)
                        run_file_line = f"{qid} Q0 {doc_id} {rank} {score} {run_id} {task_id}\n"
                        run_file.write(run_file_line)
            print(f"Retriever: Run file saved to {run_file_path}")

            # Store raw retrieved candidates for downstream applications like UniRAG
            if retrieval_config.raw_retrieval and False:
                queries_path = os.path.join(
                    mbeir_data_dir,
                    query_dir_name,
                    f"{split}/mbeir_{dataset_name}_{split}.jsonl",
                )
                candidates_path = os.path.join(
                    mbeir_data_dir, candidate_dir_name, f"mbeir_{cand_pool_name}_{split}_cand_pool.jsonl"
                )
                # When retrieving complement candidates, we want to find the closest image to a candidate caption or vica versa to have image-text pairs.
                # "MSCOCO" dataset supports both image->text and text->image queries.
                dataset_name = "MSCOCO"
                complement_retriever = (
                    InteractiveRetriever(cand_index_path, candidates_path, dataset_name, query_embedder_config)
                    if retrieval_config.retrieve_image_text_pairs
                    else None
                )
                retrieved_dict = get_raw_retrieved_candidates(
                    queries_path, candidates_path, retrieved_indices, hashed_query_ids, complement_retriever
                )
                retrieved_file_name = f"{run_id}_retrieved.jsonl"
                retrieved_file_path = os.path.join(exp_retrieved_cands_dir, retrieved_file_name)
                with open(retrieved_file_path, "w") as retrieved_file:
                    for _, v in retrieved_dict.items():
                        json.dump(v, retrieved_file)
                        retrieved_file.write("\n")
                print(f"Retriever: Retrieved file saved to {retrieved_file_path}")

            # Compute Recall@k
            recall_values_by_task = defaultdict(lambda: defaultdict(list))
            for i, retrieved_indices_for_qid in enumerate(retrieved_indices):
                # Map the retrieved FAISS indices to the original mbeir_data_ids
                retrieved_indices_for_qid = [unhash_did(idx) for idx in retrieved_indices_for_qid]
                qid = unhash_qid(hashed_query_ids[i])
                relevant_docs = qrel[qid]
                task_id = qid_to_taskid[qid]

                # Compute Recall@k for each metric
                for metric in metric_recall_list:
                    k = int(metric.split("@")[1])
                    recall_at_k = compute_recall_at_k(relevant_docs, retrieved_indices_for_qid, k)
                    recall_values_by_task[task_id][metric].append(recall_at_k)

            for task_id, recalls in recall_values_by_task.items():
                task_name = get_mbeir_task_name(int(task_id))
                result = {
                    "TaskID": int(task_id),
                    "Task": task_name,
                    "Dataset": dataset_name,
                    "Split": split,
                    "CandPool": cand_pool_name,
                }
                for metric in metric_recall_list:
                    mean_recall_at_k = round(sum(recalls[metric]) / len(recalls[metric]), 4)
                    result[metric] = mean_recall_at_k
                    print(f"Retriever: Mean {metric}: {mean_recall_at_k}")
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
    available_recall_metrics = [
        "Recall@1",
        "Recall@5",
        "Recall@10",
        "Recall@20",
        "Recall@50",
    ]
    for result in eval_results_sorted:
        key = (result["TaskID"], result["Task"], result["Dataset"], result["Split"])
        for metric in available_recall_metrics:
            grouped_results[key][result["CandPool"]].update({metric: result.get(metric, None)})

    # Write the sorted data to TSV
    if retrieval_config.write_to_tsv:
        # TODO: create a better file name
        date_time = datetime.now().strftime("%m-%d-%H")
        tsv_file_name = f"eval_results_{date_time}.tsv"
        tsv_file_path = os.path.join(exp_tsv_results_dir, tsv_file_name)
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


def run_hard_negative_mining(config):
    uniir_dir = config.uniir_dir
    mbeir_data_dir = config.mbeir_data_dir
    retrieval_config = config.retrieval_config
    expt_dir_name = config.experiment.path_suffix
    embed_dir_name = retrieval_config.embed_dir_name
    index_dir_name = retrieval_config.index_dir_name
    hard_negs_dir_name = retrieval_config.hard_negs_dir_name

    # Query data file name
    retrieval_train_dataset_config = retrieval_config.train_datasets_config
    assert retrieval_train_dataset_config.enable_retrieve, "Hard negative mining is not enabled for training data"
    dataset_name = retrieval_train_dataset_config.datasets_name[
        0
    ].lower()  # Only extract hard negatives for the first dataset
    dataset_split_name = "train"
    # Load query data
    query_data_path = os.path.join(mbeir_data_dir, "train", f"mbeir_{dataset_name}_{dataset_split_name}.jsonl")
    query_data_list = load_jsonl_as_list(query_data_path)

    # Load query IDs
    dataset_embed_dir = os.path.join(uniir_dir, embed_dir_name, expt_dir_name, dataset_split_name)
    embed_data_id_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{dataset_split_name}_ids.npy")
    query_ids = np.load(embed_data_id_path)

    # Load query embeddings
    embed_data_path = os.path.join(dataset_embed_dir, f"mbeir_{dataset_name}_{dataset_split_name}_embed.npy")

    # Load the candidate pool index
    cand_pool_name = retrieval_train_dataset_config.correspond_cand_pools_name[
        0
    ].lower()  # Only extract the first candidate pool
    cand_pool_split_name = "cand_pool"
    cand_index_dir = os.path.join(uniir_dir, index_dir_name, expt_dir_name, cand_pool_split_name)
    cand_index_path = os.path.join(cand_index_dir, f"mbeir_{cand_pool_name}_{cand_pool_split_name}.index")

    # Pretty Print dataset to perform hard negative mining
    print("-" * 30)
    print(f"Hard Negative mining, Datasets: {dataset_name}, Candidate Pools: {cand_pool_name}")
    print("-" * 30)

    # Search the index
    num_hard_negs = retrieval_config.num_hard_negs
    k = retrieval_config.k
    _, retrieved_indices = search_index(
        embed_data_path,
        cand_index_path,
        batch_size=query_ids.shape[0],
        num_cand_to_retrieve=k,
    )  # nested list of (number_of_queries, k)
    assert len(query_ids) == len(retrieved_indices)

    # Add hard negatives to the query data
    for i, query_id in enumerate(query_ids):
        query_data = query_data_list[i]
        query_id = unhash_qid(query_id)
        assert query_id == query_data["qid"]
        retrieved_indices_for_qid = retrieved_indices[i]
        retrieved_indices_for_qid = [unhash_did(idx) for idx in retrieved_indices_for_qid]

        pos_cand_list = query_data["pos_cand_list"]
        neg_cand_list = query_data["neg_cand_list"]

        # Add hard negatives to the query data
        hard_negatives = [
            doc_id
            for doc_id in retrieved_indices_for_qid
            if doc_id not in pos_cand_list and doc_id not in neg_cand_list
        ]

        # Ensure that hard_negatives has a minimum length of num_hard_negs
        if 0 < len(hard_negatives) < num_hard_negs:
            multiplier = num_hard_negs // len(hard_negatives)
            remainder = num_hard_negs % len(hard_negatives)
            hard_negatives = hard_negatives * multiplier + hard_negatives[:remainder]
        elif len(hard_negatives) == 0:
            print("Warning: hard_negatives list is empty.")

        # Truncate hard_negatives to a maximum length
        hard_negatives = hard_negatives[:num_hard_negs]
        query_data["neg_cand_list"].extend(hard_negatives)

    # Save the query data with hard negatives
    query_data_with_hard_negs_path = os.path.join(
        mbeir_data_dir,
        "train",
        hard_negs_dir_name,
        f"mbeir_{dataset_name}_hard_negs_{dataset_split_name}.jsonl",
    )  # mbeir_data_dir/train/hard_negs_dir_name/mbeir_agg_hard_negs_train.jsonl
    os.makedirs(os.path.dirname(query_data_with_hard_negs_path), exist_ok=True)
    save_list_as_jsonl(query_data_list, query_data_with_hard_negs_path)

    # Print statistics
    total_entries, _data = count_entries_in_file(query_data_with_hard_negs_path)
    print(f"MBEIR Train Data with Hard Negatives saved to {query_data_with_hard_negs_path}")
    print(f"Total number of entries in {query_data_with_hard_negs_path}: {total_entries}")
    cand_pool_path = os.path.join(
        mbeir_data_dir,
        cand_pool_split_name,
        f"mbeir_{cand_pool_name}_{cand_pool_split_name}.jsonl",
    )
    cand_pool = load_mbeir_format_pool_file_as_dict(cand_pool_path, doc_key_to_content=True, key_type="did")
    print_mbeir_format_dataset_stats(_data, cand_pool)

## step 3
def parse_arguments():
    parser = argparse.ArgumentParser(description="FAISS Pipeline")
    parser.add_argument("--uniir_dir", type=str, default="retrieval/UniIR")
    parser.add_argument("--mbeir_data_dir", type=str, default="retrieval/TIGER-Lab/M-BEIR/")
    parser.add_argument("--config_path", default="src/models/uniir_clip/clip_scorefusion/configs_scripts/large/eval/unirag/retrieval_ori.yaml", help="Path to the config file.")
    parser.add_argument(
        "--query_embedder_config_path",
        default="src/models/uniir_clip/clip_scorefusion/configs_scripts/large/eval/unirag/embed.yaml",
        help="Path to the query embedder config file. Used when retrieving candidates with complement modalities in raw_retrieval mode.",
    )
    parser.add_argument("--enable_create_index", action="store_true", help="Enable create index")
    parser.add_argument(
        "--enable_hard_negative_mining",
        action="store_true",
        help="Enable hard negative mining",
    )
    parser.add_argument("--enable_retrieval", action="store_true", default=True, help="Enable retrieval")
    return parser.parse_args()



def main():
    args = parse_arguments()
    config = OmegaConf.load(args.config_path)
    config.uniir_dir = args.uniir_dir
    config.mbeir_data_dir = args.mbeir_data_dir

    print(OmegaConf.to_yaml(config, sort_keys=False))

    interactive_retrieval = True if args.query_embedder_config_path else False
    if interactive_retrieval:
        query_embedder_config = OmegaConf.load(args.query_embedder_config_path)
        query_embedder_config.uniir_dir = args.uniir_dir
        query_embedder_config.mbeir_data_dir = args.mbeir_data_dir
        args.dist_url = query_embedder_config.dist_config.dist_url  
        dist_utils.init_distributed_mode(args)
        query_embedder_config.dist_config.gpu_id = 0
        query_embedder_config.dist_config.distributed_mode = False
    else:
        query_embedder_config = None
    if args.enable_hard_negative_mining:
        run_hard_negative_mining(config)

    if args.enable_create_index:
        create_index(config)

    if args.enable_retrieval:
        run_retrieval(config, query_embedder_config)

    if interactive_retrieval and query_embedder_config.dist_config.distributed_mode:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

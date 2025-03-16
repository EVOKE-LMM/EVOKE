"""
fashion200k_data_preprocessor.py

Module description:
    1. Convert all image files to the JPG format and resize the smaller dimension to 256.
    2. Generate candidate pool.
    3. Convert the dataset to the MBEIR format.
"""

import os
import torch
import json
import requests
from PIL import Image
from io import BytesIO
import random
import argparse
from pathlib import Path
import csv
from multiprocessing import Pool, cpu_count, Manager, Lock

from utils import (
    resize_and_convert_image_to_jpg,
    is_valid_image,
    get_dataset_id,
    format_string,
    count_entries_in_file,
    generate_mbeir_format_doc_key,
    load_mbeir_format_pool_file_as_dict,
    print_mbeir_format_cand_pool_stats,
    save_list_as_jsonl,
    load_jsonl_as_list,
    print_mbeir_format_dataset_stats,
    parallel_process_image_directory,
    aggregate_candidates_for_mbeir_format_dataset,
)

FASHION200K_QUERY_MODALITY_IMAGE = "image"
FASHION200K_QUERY_MODALITY_TEXT = "text"
FASHION200K_CANDIDATE_MODALITY_IMAGE = "image"
FASHION200K_CANDIDATE_MODALITY_TEXT = "text"
FASHION200K_DATASET_ID = get_dataset_id("Fashion200K")
assert FASHION200K_DATASET_ID is not None, "Unknown dataset name!"


def fashion200k_to_mbeir_entry(
    fashion200k_entry,
    candidate_pool,
    mbeir_data_dir,
):
    """
    Convert fashion200k data format to MBEIR format.
    Sample MBEIR entry:
    {
    "qid": "0:2",
    "query_txt": null,
    "query_img_path": "...jpg",
    "query_modality": "...",
    "query_src_content": "..."
    "pos_cand_list":
        [{
            "did": "0:127542",
            "txt": "...",
            "img_path": null,
            "modality": "text"
            "src_content": {"..._entry": ..._entry}
        }
        ...]
    "neg_cand_list":
        [{
            "did": "0:127542",
            "txt": "...",
            "img_path": null,
            "modality": "text"
            "src_content": {"..._entry": ..._entry}
        }
        ...]
    }
    """
    # Process image path and description
    img_path = fashion200k_entry["img_path"]
    path_parts = img_path.split("/")
    base_filename, _ = os.path.splitext("/".join(path_parts[1:]))  # Removing 'women/' and '.jpeg'
    img_path = os.path.join("mbeir_images", "fashion200k_images", base_filename + ".jpg")

    txt = format_string(fashion200k_entry["txt"])

    # Generate Image2Text MBEIR entry
    query_img_path = img_path
    if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
        print(f"Warning: Invalid query_img_path: {query_img_path}")
        return None  # query image is missing
    mbeir_entry_img2txt = {
        "qid": None,
        "query_txt": None,
        "query_img_path": query_img_path,
        "query_modality": FASHION200K_QUERY_MODALITY_IMAGE,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    # Add positive text candidate
    doc_key = "-".join([txt, FASHION200K_CANDIDATE_MODALITY_TEXT])
    pos_candidate_did = candidate_pool.get(doc_key, None)
    if pos_candidate_did:
        mbeir_entry_img2txt["pos_cand_list"].append(pos_candidate_did)
    else:
        print(f"Warning: No positive candidate for query_img_path {mbeir_entry_img2txt['query_img_path']}")
        return None

    # Generate Text2Image MBEIR entry
    mbeir_entry_txt2img = {
        "qid": None,
        "query_txt": txt,
        "query_img_path": None,
        "query_modality": FASHION200K_QUERY_MODALITY_TEXT,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    # Add positive image candidate
    doc_key = "-".join([img_path, FASHION200K_CANDIDATE_MODALITY_IMAGE])
    pos_candidate_did = candidate_pool.get(doc_key, None)
    if pos_candidate_did:
        mbeir_entry_txt2img["pos_cand_list"].append(pos_candidate_did)
    else:
        print(f"Warning: No positive candidate for query_txt {mbeir_entry_txt2img['query_txt']}")
        return None

    mbeir_entries = [mbeir_entry_img2txt, mbeir_entry_txt2img]
    return mbeir_entries


def get_deduplicated_fashion200k_data(fashion200k_data):
    deduplicated_data = {}
    for fashion200k_entry in fashion200k_data:
        data_id = fashion200k_entry["img_path"]
        if data_id not in deduplicated_data:
            deduplicated_data[data_id] = fashion200k_entry
        else:
            print(f"\n Warning: Duplicate data entry: {data_id}")
            print(
                f"fashion200k_entry: {fashion200k_entry} and deduplicated_data[data_id]: {deduplicated_data[data_id]}"
            )

    # Convert the dictionary values into a list
    return list(deduplicated_data.values())


def fashion200k_to_mbeir(fashion200k_data, candidate_pool_file_path, mbeir_data_dir):
    """
    fashion200k dataset to MBEIR format.
    """
    mbeir_entries_merged = []

    # Load candidate pool
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(
        candidate_pool_file_path,
        doc_key_to_content=False,
        key_type="mbeir_converted_key",
    )
    # We can keep this if we don't want multiple captions mapped to the same image.
    fashion200k_data = get_deduplicated_fashion200k_data(fashion200k_data)

    for fashion200k_entry in fashion200k_data:
        mbeir_entries = fashion200k_to_mbeir_entry(
            fashion200k_entry,
            cand_pool_dict,
            mbeir_data_dir,
        )
        if mbeir_entries:  # Skip invalid entries
            mbeir_entries_merged.extend(mbeir_entries)
    return mbeir_entries_merged


def generate_fashion200k_candidate_pool(
    fashion200k_labels_dir,
    fashion200k_cand_pool_path,
    mbeir_data_dir,
):
    """
    Generate fashion200k candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure of the fashion200k_images_dir:
    fashion200k_images/
        ├── dresses
        │    ├── casual_and_day_dresses
        │    │   ├── xxx.jpg
        │    │   ├── ...
        │    ├── cocktail_dresses
        │    ├── ...
        ├── jackets
        ├── ...
    """

    # Collect all label files in the folder
    label_files = [
        os.path.join(fashion200k_labels_dir, filename)
        for filename in os.listdir(fashion200k_labels_dir)
        if filename.endswith(".txt")
    ]

    document_id = 1  # Note: We start from 1 for document IDs
    seen_txts = set()  # To store description that we've already seen
    seen_image_paths = set()  # To store image paths that we've already seen

    # Open separate files for text and image entries
    with open(fashion200k_cand_pool_path, "w") as outfile:
        for label_file in label_files:
            with open(label_file, "r") as source:
                for line in source:
                    img_path, _, description = line.strip().split("\t")
                    # Note: we always store relative paths to MBEIR data directory
                    path_parts = img_path.split("/")
                    base_filename, _ = os.path.splitext("/".join(path_parts[1:]))  # Removing 'women/' and '.jpeg'
                    img_path = os.path.join("mbeir_images", "fashion200k_images", base_filename + ".jpg")
                    description = format_string(description)

                    # Track if we've seen both the caption and image path
                    seen_both = description in seen_txts and img_path in seen_image_paths

                    if not seen_both:
                        # If description hasn't been seen, create text entry
                        if description not in seen_txts:
                            # If the description is empty, skip it
                            if not description:
                                print(f"Warning: Empty description: {img_path}")
                            else:
                                candidate_pool_entry_txt = {
                                    "txt": description,
                                    "img_path": None,
                                    "modality": FASHION200K_CANDIDATE_MODALITY_TEXT,
                                    "did": f"{FASHION200K_DATASET_ID}:{document_id}",
                                    "src_content": None,
                                }
                                document_id += 1  # increment for next entry
                                outfile.write(json.dumps(candidate_pool_entry_txt) + "\n")
                                seen_txts.add(description)

                        # If image path hasn't been seen, create image entry
                        if img_path not in seen_image_paths:
                            # If image is not valid, skip it
                            if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                                print(f"Warning: Invalid image: {img_path}")
                            else:
                                candidate_pool_entry_img = {
                                    "txt": None,
                                    "img_path": img_path,
                                    "modality": FASHION200K_CANDIDATE_MODALITY_IMAGE,
                                    "did": f"{FASHION200K_DATASET_ID}:{document_id}",
                                    "src_content": None,
                                }
                                document_id += 1  # increment for next entry
                                outfile.write(json.dumps(candidate_pool_entry_img) + "\n")
                                seen_image_paths.add(img_path)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format fashion200k images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--fashion200k_images_dir",
        type=str,
        default="mbeir_images/fashion200k_images/",
        help="Relative directory path to save fashion200k images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--fashion200k_dir",
        type=str,
        default="src_data/fashion200k",
        help="Relative directory path of fashion200k files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating fashion200k candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting fashion200k data to MBEIR format.",
    )
    parser.add_argument(
        "--trim_train_data",
        action="store_true",
        help="Trim the training data queries.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--split_candidate_pool_by_task",
        action="store_true",
        help="Enable splitting the candidate pool according to task.",
    )
    parser.add_argument(
        "--generate_validation_data",
        action="store_true",
        help="Enable generating validation data.",
    )
    parser.add_argument(
        "--split_query_data_by_task",
        action="store_true",
        help="Enable splitting the query data according to task.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the fashion200k dataset
    # So all the paths are hardcoded.
    fashion200k_dir = os.path.join(args.mbeir_data_dir, args.fashion200k_dir)
    fashion200k_images_dir = os.path.join(args.mbeir_data_dir, args.fashion200k_images_dir)
    fashion200k_labels_dir = os.path.join(fashion200k_dir, "labels")
    fashion200k_candidate_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_cand_pool.jsonl")

    if args.enable_image_processing:
        print(f"Processing images in {fashion200k_images_dir}...")
        parallel_process_image_directory(fashion200k_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating fashion200k candidate pool in mbeir format...")
        generate_fashion200k_candidate_pool(
            fashion200k_labels_dir,
            fashion200k_candidate_pool_path,
            args.mbeir_data_dir,
        )
        print(f"Candidate pool saved to {fashion200k_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(fashion200k_candidate_pool_path)

    # Convert fashion200k data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting fashion200k data to MBEIR format...")

        # Helper function to load fashion200k data from the text files
        def load_fashion200k_data_from_txt(txt_path):
            with open(txt_path, "r", encoding="utf-8") as file:
                data = []
                for line in file:
                    img_path, _, description = line.strip().split("\t")
                    # Assuming you need to store img_path and description as in previous script
                    data.append({"img_path": img_path, "txt": description})
                return data

        splits = ["train", "test"]
        split_data = {split: [] for split in splits}

        # Define the paths to the text files based on naming pattern
        types = ["dress", "jacket", "pants", "skirt", "top"]

        for type in types:
            for split in splits:
                file_path = os.path.join(fashion200k_dir, "labels", f"{type}_{split}_detect_all.txt")
                data = load_fashion200k_data_from_txt(file_path)
                split_data[split].extend(data)

        for split, fashion200k_data_split in split_data.items():
            mbeir_format_fashion200k_data_path = os.path.join(fashion200k_dir, f"mbeir_fashion200k_{split}.jsonl")
            mbeir_entries = fashion200k_to_mbeir(
                fashion200k_data_split,
                fashion200k_candidate_pool_path,
                args.mbeir_data_dir,
            )

            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries, print_duplicate=False)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{FASHION200K_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_fashion200k_data_path)

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_fashion200k_data_path)
            print(f"MBEIR format fashion200k {split} data saved to {mbeir_format_fashion200k_data_path}")
            print(f"Total number of entries in {mbeir_format_fashion200k_data_path}: {total_entries}")
            fashion200k_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                fashion200k_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, fashion200k_cand_pool_dict)

    # Trim the training data queries to 30K
    if args.trim_train_data:
        trim_num = 15000
        print(f" Trim the training data queries to {2*trim_num}...")
        fashion200k_train_data_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_train.jsonl")
        fashion200k_train_data_trimmed_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_train_trimmed.jsonl")
        fashion200k_train_data = load_jsonl_as_list(fashion200k_train_data_path)
        txt2img_entries = []
        img2txt_entries = []
        for entry in fashion200k_train_data:
            if entry["query_modality"] == "text":
                txt2img_entries.append(entry)
            else:
                img2txt_entries.append(entry)
        random.seed(2023)
        random.shuffle(txt2img_entries)
        random.shuffle(img2txt_entries)
        fashion200k_train_data_trimmed = txt2img_entries[:trim_num] + img2txt_entries[:trim_num]
        random.shuffle(fashion200k_train_data_trimmed)

        # Reassign query IDs
        for i, entry in enumerate(fashion200k_train_data_trimmed):
            entry.update({"qid": f"{FASHION200K_DATASET_ID}:{i + 1}"})
        save_list_as_jsonl(fashion200k_train_data_trimmed, fashion200k_train_data_trimmed_path)

        total_entries, _data = count_entries_in_file(fashion200k_train_data_trimmed_path)
        print(f"Trimmed fashion200k train data saved to {fashion200k_train_data_trimmed_path}")
        print(f"Total number of entries in {fashion200k_train_data_trimmed_path}: {total_entries}")
        fashion200k_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            fashion200k_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(_data, fashion200k_cand_pool_dict)

        # Rename
        os.rename(
            fashion200k_train_data_path,
            os.path.join(fashion200k_dir, "mbeir_fashion200k_train_untrimmed.jsonl"),
        )
        print(
            f"Renamed {fashion200k_train_data_path} to {os.path.join(fashion200k_dir, 'mbeir_fashion200k_train_untrimmed.jsonl')}"
        )
        os.rename(fashion200k_train_data_trimmed_path, fashion200k_train_data_path)
        print(f"Renamed {fashion200k_train_data_trimmed_path} to {fashion200k_train_data_path}")

    # Split the cand pool according to task
    if args.split_candidate_pool_by_task:
        print("Split the candidate pool according to task")

        # Load the candidate pool
        fashion200k_cand_pool = load_jsonl_as_list(fashion200k_candidate_pool_path)

        # Split the candidate pool
        fashion200k_task0_cand_pool = []
        fashion200k_task3_cand_pool = []
        for fashion200k_cand in fashion200k_cand_pool:
            if fashion200k_cand["modality"] == "image":
                fashion200k_task0_cand_pool.append(fashion200k_cand)
            elif fashion200k_cand["modality"] == "text":
                fashion200k_task3_cand_pool.append(fashion200k_cand)
            else:
                raise ValueError(f"Unknown modality: {fashion200k_cand['modality']}")
        print(f"Number of candidates for task 0: {len(fashion200k_task0_cand_pool)}")
        print(f"Number of candidates for task 3: {len(fashion200k_task3_cand_pool)}")

        # Save the candidate pool
        fashion200k_task0_cand_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_task0_cand_pool.jsonl")
        fashion200k_task3_cand_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_task3_cand_pool.jsonl")
        save_list_as_jsonl(fashion200k_task0_cand_pool, fashion200k_task0_cand_pool_path)
        save_list_as_jsonl(fashion200k_task3_cand_pool, fashion200k_task3_cand_pool_path)
        print(f"Saved task 0 candidate pool to {fashion200k_task0_cand_pool_path}")
        print(f"Saved task 3 candidate pool to {fashion200k_task3_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(fashion200k_task0_cand_pool_path)
        print_mbeir_format_cand_pool_stats(fashion200k_task3_cand_pool_path)

    # Generate validation data
    if args.generate_validation_data:
        # We equally split the test data into validation and test data
        print("Generating validation data...")
        fashion200k_test_data_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_test.jsonl")
        fashion200k_val_data_path_after_split = os.path.join(fashion200k_dir, "mbeir_fashion200k_val_after_split.jsonl")
        fashion200k_test_data_path_after_split = os.path.join(
            fashion200k_dir, "mbeir_fashion200k_test_after_split.jsonl"
        )

        # Load the test data
        fashion200k_test_data_before_split = load_jsonl_as_list(fashion200k_test_data_path)
        # trim the size of the test data by half
        fashion200k_test_data_before_split = fashion200k_test_data_before_split[
            : len(fashion200k_test_data_before_split) // 2
        ]
        random.seed(2023)
        random.shuffle(fashion200k_test_data_before_split)
        fashion200k_val_data = fashion200k_test_data_before_split[: len(fashion200k_test_data_before_split) // 3]
        fashion200k_test_data = fashion200k_test_data_before_split[len(fashion200k_test_data_before_split) // 3 * 2 :]
        save_list_as_jsonl(fashion200k_val_data, fashion200k_val_data_path_after_split)
        save_list_as_jsonl(fashion200k_test_data, fashion200k_test_data_path_after_split)
        fashion200k_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            fashion200k_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print(f"Saved validation data to {fashion200k_val_data_path_after_split}")
        print_mbeir_format_dataset_stats(fashion200k_val_data, fashion200k_cand_pool_dict)
        print(f"Saved test data to {fashion200k_test_data_path_after_split}")
        print_mbeir_format_dataset_stats(fashion200k_test_data, fashion200k_cand_pool_dict)

    # Split the query data according to task
    if args.split_query_data_by_task:
        print("Split the query data according to task")
        fashion200k_task0_cand_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_task0_cand_pool.jsonl")
        fashion200k_task3_cand_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_task3_cand_pool.jsonl")

        for split in ["val", "test"]:
            data_path = os.path.join(fashion200k_dir, f"mbeir_fashion200k_{split}_after_split.jsonl")
            task0_data_path = os.path.join(fashion200k_dir, f"mbeir_fashion200k_task0_{split}.jsonl")
            task3_data_path = os.path.join(fashion200k_dir, f"mbeir_fashion200k_task3_{split}.jsonl")

            # Load the data
            fashion200k_data = load_jsonl_as_list(data_path)
            task0_data = []
            task3_data = []
            for entry in fashion200k_data:
                if entry["query_modality"] == "text":
                    task0_data.append(entry)
                elif entry["query_modality"] == "image":
                    task3_data.append(entry)
                else:
                    raise ValueError(f"Unknown modality: {entry['query_modality']}")

            # Save the data
            save_list_as_jsonl(task0_data, task0_data_path)
            save_list_as_jsonl(task3_data, task3_data_path)
            print(f"Saved task 0 data to {task0_data_path}")
            fashion200k_task0_cand_pool = load_mbeir_format_pool_file_as_dict(
                fashion200k_task0_cand_pool_path,
                doc_key_to_content=True,
                key_type="did",
            )
            print_mbeir_format_dataset_stats(task0_data, fashion200k_task0_cand_pool)
            print(f"Saved task 3 data to {task3_data_path}")
            fashion200k_task3_cand_pool = load_mbeir_format_pool_file_as_dict(
                fashion200k_task3_cand_pool_path,
                doc_key_to_content=True,
                key_type="did",
            )
            print_mbeir_format_dataset_stats(task3_data, fashion200k_task3_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        fashion200k_train_candidate_pool_path = os.path.join(fashion200k_dir, "mbeir_fashion200k_train_cand_pool.jsonl")
        mbeir_format_fashion200k_train_data_path = os.path.join(fashion200k_dir, f"mbeir_fashion200k_train.jsonl")
        assert os.path.exists(
            mbeir_format_fashion200k_train_data_path
        ), f"File {mbeir_format_fashion200k_train_data_path} does not exist"

        # Load the training data
        fashion200k_train_candidate_pool = {}
        fashion200k_cand_pool = load_mbeir_format_pool_file_as_dict(
            fashion200k_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_fashion200k_train_data = load_jsonl_as_list(mbeir_format_fashion200k_train_data_path)
        for entry in mbeir_format_fashion200k_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = fashion200k_cand_pool[did]
                if did not in fashion200k_train_candidate_pool:
                    fashion200k_train_candidate_pool[did] = cand
                else:
                    if fashion200k_train_candidate_pool[did] != cand:
                        print(
                            f"Duplicate did for two candidates found: {fashion200k_train_candidate_pool[did]} and {cand}"
                        )

        # Save the training candidate pool
        fashion200k_train_candidate_pool_list = list(fashion200k_train_candidate_pool.values())
        fashion200k_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(fashion200k_train_candidate_pool_list, fashion200k_train_candidate_pool_path)
        print(f"Saved training candidate pool to {fashion200k_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(fashion200k_train_candidate_pool_path)


if __name__ == "__main__":
    main()

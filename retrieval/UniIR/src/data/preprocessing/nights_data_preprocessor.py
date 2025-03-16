"""
nights_data_preprocessor.py

Module description:
    1. Convert all image files to the JPG format and resize the smaller dimension to 256.
    2. Generate candidate pool.
    3. Convert the dataset to the MBEIR format.
"""

import os
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
    print_mbeir_format_cand_pool_stats,
    save_list_as_jsonl,
    load_jsonl_as_list,
    print_mbeir_format_dataset_stats,
    parallel_process_image_directory,
    aggregate_candidates_for_mbeir_format_dataset,
    load_mbeir_format_pool_file_as_dict,
)

NIGHTS_QUERY_MODALITY = "image"
NIGHTS_CANDIDATE_MODALITY = "image"
NIGHTS_DATASET_ID = get_dataset_id("NIGHTS")
assert NIGHTS_DATASET_ID is not None, "Unknown dataset name!"


def nights_to_mbeir_entry(
    nights_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert nights data format to MBEIR format.
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
    dataset_id = get_dataset_id("NIGHTS")
    assert dataset_id is not None, "Unknown dataset name!"

    query_img_name = os.path.splitext(nights_entry["ref_path"])[0] + ".jpg"
    query_img_path = os.path.join("mbeir_images", "nights_images", query_img_name)
    if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
        print(f"Warning: Invalid query_img_path: {query_img_path}")
        return None  # query image is missing

    mbeir_entry = {
        "qid": None,
        "query_txt": None,
        "query_img_path": query_img_path,
        "query_modality": NIGHTS_QUERY_MODALITY,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    if include_src_content:
        query_src_content = {
            "id": nights_entry["id"],
            "target": nights_entry["right_vote"],
            "split": nights_entry["split"],
            "is_imagenet": nights_entry["is_imagenet"],
        }
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)

    # Add candidates
    def get_key_from_path(img_path):
        _, tail = os.path.split(os.path.split(img_path)[0])  # Split twice to get the '000' folder name
        filename = os.path.splitext(os.path.basename(img_path))[0]  # Get filename without extension
        return os.path.join(tail, filename)

    # For positive candidate and negative candidate
    if nights_entry["right_vote"] == "0":
        pos_key = get_key_from_path(nights_entry["left_path"])
        neg_key = get_key_from_path(nights_entry["right_path"])
    elif nights_entry["right_vote"] == "1":
        pos_key = get_key_from_path(nights_entry["right_path"])
        neg_key = get_key_from_path(nights_entry["left_path"])
    else:
        raise ValueError(f"Invalid right_vote value: {nights_entry['right_vote']}")
    assert pos_key is not None and neg_key is not None, f"Invalid paths: {nights_entry}"

    pos_candidate = candidate_pool.get(pos_key, None)
    if pos_candidate:
        mbeir_entry["pos_cand_list"].append(pos_candidate["did"])
    else:
        print(f"Warning: No positive candidate for reference {nights_entry['reference']}")
        return None

    neg_candidate = candidate_pool.get(neg_key, None)
    if neg_candidate:
        mbeir_entry["neg_cand_list"].append(neg_candidate["did"])
    else:
        print(f"Warning: No negative candidate for reference {nights_entry['reference']}")
        return None
    return mbeir_entry


def load_mbeir_format_nights_pool_file_as_dict(pool_file_path):
    """
    Load the nights candidate pool file into a dictionary.
    """
    pool_dict = {}
    assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."
    with open(pool_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            src_content = json.loads(entry["src_content"])  # Load the src_content as a dictionary
            doc_key = src_content["image_id"]  # Use the image name as the key
            assert doc_key not in pool_dict, f"Duplicate key: {doc_key}"
            pool_dict[doc_key] = entry
    return pool_dict


def get_deduplicated_nights_data(nights_data):
    deduplicated_data = {}
    for nights_entry in nights_data:
        data_id = nights_entry["ref_path"]
        if data_id not in deduplicated_data:
            deduplicated_data[data_id] = nights_entry
        else:
            print(f"\n Warning: Duplicate data entry: {data_id}")
            print(f"nights_entry: {nights_entry}")

    # Convert the dictionary values into a list
    return list(deduplicated_data.values())


def nights_to_mbeir(nights_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    """
    nights dataset to MBEIR format.
    """
    mbeir_entries = []

    # Load candidate pool
    candidate_pool = load_mbeir_format_nights_pool_file_as_dict(candidate_pool_file_path)
    # nights_data = get_deduplicated_nights_data(nights_data)

    for nights_entry in nights_data:
        mbeir_entry = nights_to_mbeir_entry(
            nights_entry,
            candidate_pool,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entry:  # Skip invalid entries
            mbeir_entries.append(mbeir_entry)
    return mbeir_entries


def generate_nights_candidate_pool(
    nights_distort_images_dir,
    nights_candidate_pool_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate NIGHTS candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure of the nights_images_dir:
        ├── nights_distort_images_dir/
        │   ├── 000/
        │   │   ├── 000_0.jpg
        │   │   ├── 000_1.jpg
        │   │   ├── ...
        │   │   ├── 999_0.jpg
        │   │   └── 999_1.jpg
        │   └──  001/ 002/ ... 099/
    """

    # Create the image_name_set by iterating over subdirectories and listing all files with .jpg extension.
    image_name_set = set()
    for subdir in os.listdir(nights_distort_images_dir):
        subdir_path = os.path.join(nights_distort_images_dir, subdir)
        if os.path.isdir(subdir_path):
            for fname in os.listdir(subdir_path):
                if fname.endswith(".jpg"):
                    image_name_set.add(os.path.join(subdir, fname))

    document_id = 1  # Note: We start from 1 for document IDs

    with open(nights_candidate_pool_path, "w") as outfile:
        for image_name in image_name_set:
            # Note: we always store relative paths to MBEIR data directory
            dir_name = os.path.basename(nights_distort_images_dir)
            img_path_rel = os.path.join("mbeir_images", "nights_images", dir_name, image_name)
            img_path_abs = os.path.join(mbeir_data_dir, img_path_rel)

            # if the image is valid, add it to the candidate pool
            if is_valid_image(img_path_abs):
                candidate_pool_entry = {
                    "txt": None,
                    "img_path": img_path_rel,
                    "modality": NIGHTS_CANDIDATE_MODALITY,
                    "did": f"{NIGHTS_DATASET_ID}:{document_id}",
                }
                if include_src_content:
                    src_content = {
                        "image_id": os.path.splitext(image_name)[0],
                    }
                    candidate_pool_entry["src_content"] = json.dumps(src_content)
                document_id += 1  # increment for next entry
                outfile.write(json.dumps(candidate_pool_entry) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format nights images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--nights_images_dir",
        type=str,
        default="mbeir_images/nights_images/",
        help="Relative directory path to save nights images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--nights_dir",
        type=str,
        default="src_data/nights",
        help="Relative directory path of nights files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating nights candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting nights data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the NIGHTS dataset
    # So all the paths are hardcoded.
    nights_dir = os.path.join(args.mbeir_data_dir, args.nights_dir)
    nights_images_dir = os.path.join(args.mbeir_data_dir, args.nights_images_dir)
    nights_distort_images_dir = os.path.join(nights_images_dir, "distort")
    nights_candidate_pool_path = os.path.join(nights_dir, "mbeir_nights_cand_pool.jsonl")
    nights_data_csv_path = os.path.join(nights_dir, "data.csv")
    nights_captions_dir = os.path.join(nights_dir, "captions")

    if args.enable_image_processing:
        print(f"Processing images in {nights_images_dir}...")
        parallel_process_image_directory(nights_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating nights candidate pool in mbeir format...")
        generate_nights_candidate_pool(
            nights_distort_images_dir,
            nights_candidate_pool_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"Candidate pool saved to {nights_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(nights_candidate_pool_path)

    # Convert nights data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting NIGHTS data to MBEIR format...")

        def load_nights_data_from_csv(csv_path):
            with open(csv_path, "r", encoding="utf-8") as file:
                reader = csv.DictReader(file)
                return [row for row in reader]

        # Load data from CSV
        _nights_data = load_nights_data_from_csv(nights_data_csv_path)

        # Group by the split field
        train_data = [entry for entry in _nights_data if entry["split"] == "train"]
        val_data = [entry for entry in _nights_data if entry["split"] == "val"]
        test_data = [entry for entry in _nights_data if entry["split"] == "test"]
        data_set_list = [("train", train_data), ("val", val_data), ("test", test_data)]

        for split, nights_data_split in data_set_list:
            mbeir_format_nights_data_path = os.path.join(nights_dir, f"mbeir_nights_{split}.jsonl")
            mbeir_entries = nights_to_mbeir(
                nights_data_split,
                nights_candidate_pool_path,
                args.mbeir_data_dir,
                include_src_content=True,
            )

            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries, print_duplicate=True)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{NIGHTS_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_nights_data_path)

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_nights_data_path)
            print(f"MBEIR format nights {split} data saved to {mbeir_format_nights_data_path}")
            print(f"Total number of entries in {mbeir_format_nights_data_path}: {total_entries}")
            nights_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                nights_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, nights_cand_pool_dict)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        nights_train_candidate_pool_path = os.path.join(nights_dir, "mbeir_nights_train_cand_pool.jsonl")
        mbeir_format_nights_train_data_path = os.path.join(nights_dir, f"mbeir_nights_train.jsonl")
        assert os.path.exists(
            mbeir_format_nights_train_data_path
        ), f"File {mbeir_format_nights_train_data_path} does not exist"

        # Load the training data
        nights_train_candidate_pool = {}
        nights_cand_pool = load_mbeir_format_pool_file_as_dict(
            nights_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_nights_train_data = load_jsonl_as_list(mbeir_format_nights_train_data_path)
        for entry in mbeir_format_nights_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = nights_cand_pool[did]
                if did not in nights_train_candidate_pool:
                    nights_train_candidate_pool[did] = cand
                else:
                    if nights_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {nights_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        nights_train_candidate_pool_list = list(nights_train_candidate_pool.values())
        nights_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(nights_train_candidate_pool_list, nights_train_candidate_pool_path)
        print(f"Saved training candidate pool to {nights_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(nights_train_candidate_pool_path)


if __name__ == "__main__":
    main()

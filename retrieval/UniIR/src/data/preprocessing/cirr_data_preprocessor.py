"""
cirr_data_preprocessor.py

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

CIRR_QUERY_MODALITY = "image,text"
CIRR_CANDIDATE_MODALITY = "image"
CIRR_DATASET_ID = get_dataset_id("CIRR")
assert CIRR_DATASET_ID is not None, "Unknown dataset name!"


def cirr_to_mbeir_entry(
    cirr_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert CIRR data format to MBEIR format.
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

    # Fetch query image path from candidate_pool
    # Note: store query image inside the candidate pool is fine since this is an (image, text) retrieval task.
    _query = candidate_pool.get(cirr_entry["reference"], None)
    if not _query:
        print(f"Warning: Can not fetch query image path for reference {cirr_entry['reference']}")
        return None  # query image is not in the candidate pool
    query_img_path = _query["img_path"]

    if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
        print(f"Warning: Invalid query_img_path : {query_img_path}")
        return None  # query image is missing

    query_txt = format_string(cirr_entry["caption"])
    if not query_txt:
        print(f"Warning: Invalid query_txt : {query_txt}")
        return None  # query text is missing

    mbeir_entry = {
        "qid": None,
        "query_txt": query_txt,
        "query_img_path": query_img_path,
        "query_modality": CIRR_QUERY_MODALITY,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    if include_src_content:
        query_src_content = {
            "id": str(cirr_entry.get("reference", "")),
            # "pairid_list": cirr_entry["pairid_list"],  #TODO: this field is not supported by HF Datasets
            # "img_set_list": cirr_entry["img_set_list"],
            # "target_hard_list": cirr_entry["target_hard_list"],
        }
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)

    # Add candidates
    for target, value in cirr_entry["target_soft"].items():
        candidate = candidate_pool.get(target, None)
        if not candidate:
            print(f"Warning: Can not fetch candidate pool info from target {target}")
            continue
        if value == 1.0:
            mbeir_entry["pos_cand_list"].append(candidate["did"])
        else:
            mbeir_entry["neg_cand_list"].append(candidate["did"])  # For score 0.2 0.5 -1 we use as negative

    # We need at least one positive candidate
    if len(mbeir_entry["pos_cand_list"]) == 0:
        print(f"Warning: No positive candidate for reference {cirr_entry['reference']}")
        # print(f"cirr_entry: {cirr_entry}")
        return None

    return mbeir_entry


def load_mbeir_format_cirr_pool_file_as_dict(pool_file_path):
    """
    Load the cirr candidate pool file into a dictionary.
    cirr has unique candidate IDs the image name, so we can use them as keys.
    """
    pool_dict = {}
    assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."
    with open(pool_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            src_content = json.loads(entry["src_content"])
            doc_key = src_content["id"]  # Use the image name as the key
            pool_dict[doc_key] = entry
    return pool_dict


def get_deduplicated_cirr_data(cirr_data):
    deduplicated_data = {}
    for cirr_entry in cirr_data:
        data_id = (cirr_entry["reference"], cirr_entry["caption"])

        if data_id not in deduplicated_data:
            # Initialize as lists for new entries
            deduplicated_data[data_id] = {
                **cirr_entry,
                "pairid_list": [cirr_entry["pairid"]],
                "target_hard_list": [cirr_entry["target_hard"]],
                "img_set_list": [cirr_entry["img_set"]],
            }
            # Remove the original fields
            deduplicated_data[data_id].pop("pairid", None)
            deduplicated_data[data_id].pop("target_hard", None)
            deduplicated_data[data_id].pop("img_set", None)
        else:
            previous_entry = deduplicated_data[data_id]
            # Append to lists for duplicate entries
            previous_entry["pairid_list"].append(cirr_entry["pairid"])
            previous_entry["target_hard_list"].append(cirr_entry["target_hard"])
            previous_entry["img_set_list"].append(cirr_entry["img_set"])

            # Merge 'target_soft' dictionaries (as before)
            merged_target_soft = {
                **previous_entry["target_soft"],
                **cirr_entry["target_soft"],
            }
            previous_entry["target_soft"] = merged_target_soft

            # print(f"\n Warning: Duplicate data entry: {data_id}")
            # print(f"cirr_entry: {cirr_entry}")
            # print(f"merged cirr_entry: {previous_entry}")

    # Convert the dictionary values into a list
    return list(deduplicated_data.values())


def cirr_to_mbeir(cirr_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    """
    cirr dataset to MBEIR format.
    """
    mbeir_entries = []

    # Load candidate pool
    candidate_pool = load_mbeir_format_cirr_pool_file_as_dict(candidate_pool_file_path)

    for cirr_entry in cirr_data:
        mbeir_entry = cirr_to_mbeir_entry(
            cirr_entry,
            candidate_pool,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entry:  # Skip invalid entries
            mbeir_entries.append(mbeir_entry)
    return mbeir_entries


def generate_cirr_candidate_pool(cirr_images_dir, cirr_candidate_pool_path, mbeir_data_dir, include_src_content=True):
    """
    Generate CIRR candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure of the cirr_images_dir:
    cirr_images_dir
        ├── dev
             ├── ... jpg
             ├── ... jpg
        ├── test1
            ├── ... jpg
            ├── ... jpg
        └── train
            ├── 0
                ├── ... jpg
                ├── ... jpg
            ├── 1
            ├── 10
    """

    image_paths = set()

    # Collect all image paths from dev and test1 directories
    for subdir in ["dev", "test1"]:
        image_paths.update(
            os.path.join(subdir, fname)
            for fname in os.listdir(os.path.join(cirr_images_dir, subdir))
            if fname.endswith(".jpg")
        )

    # Collect all image paths from the train subdirectories
    for subdir in os.listdir(os.path.join(cirr_images_dir, "train")):
        path = os.path.join(cirr_images_dir, "train", subdir)
        if os.path.isdir(path):
            image_paths.update(
                os.path.join("train", subdir, fname) for fname in os.listdir(path) if fname.endswith(".jpg")
            )

    document_id = 1  # Note: We start from 1 for document IDs

    with open(cirr_candidate_pool_path, "w") as outfile:
        for image_path in image_paths:
            img_path_rel = os.path.join("mbeir_images", "cirr_images", image_path)
            img_path_abs = os.path.join(mbeir_data_dir, img_path_rel)

            # If the image is valid, add it to the candidate pool
            if is_valid_image(img_path_abs):
                candidate_pool_entry = {
                    "txt": None,
                    "img_path": img_path_rel,
                    "modality": CIRR_CANDIDATE_MODALITY,
                    "did": f"{CIRR_DATASET_ID}:{document_id}",
                }
                if include_src_content:
                    src_content = {
                        "id": os.path.splitext(os.path.basename(image_path))[0],
                    }  # Cast to string to avoid JSON serialization error
                    candidate_pool_entry["src_content"] = json.dumps(src_content)
                document_id += 1  # Increment for next entry
                outfile.write(json.dumps(candidate_pool_entry) + "\n")


def generate_cirr_candidate_pool_from_splits(
    image_splits_dir, cirr_candidate_pool_path, mbeir_data_dir, include_src_content=True
):
    """
    Generate CIRR candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure:
    cirr_images_dir
        ├── dev
             ├── ... jpg
             ├── ... jpg
        ├── test1
            ├── ... jpg
            ├── ... jpg
        └── train
            ├── 0
                ├── ... jpg
                ├── ... jpg
            ├── 1
            ├── 10
    image_splits_dir
        ├── split.rc2.test1.json
        ├── split.rc2.train.json
        └── split.rc2.val.json
    """

    # Extract relevant image paths from split files
    image_paths_from_splits = set()
    for split_file in os.listdir(image_splits_dir):
        with open(os.path.join(image_splits_dir, split_file), "r") as f:
            split_data = json.load(f)
            for k, v in split_data.items():
                image_path_from_split = v.lstrip("./")
                image_path_jpg = os.path.splitext(image_path_from_split)[0] + ".jpg"
                image_paths_from_splits.add(image_path_jpg)

    document_id = 1  # Note: We start from 1 for document IDs

    with open(cirr_candidate_pool_path, "w") as outfile:
        for image_path in image_paths_from_splits:
            img_path_rel = os.path.join("mbeir_images", "cirr_images", image_path)
            img_path_abs = os.path.join(mbeir_data_dir, img_path_rel)

            # If the image is valid, add it to the candidate pool
            if is_valid_image(img_path_abs):
                candidate_pool_entry = {
                    "txt": None,
                    "img_path": img_path_rel,
                    "modality": CIRR_CANDIDATE_MODALITY,
                    "did": f"{CIRR_DATASET_ID}:{document_id}",
                }
                if include_src_content:
                    src_content = {
                        "id": os.path.splitext(os.path.basename(image_path))[0],
                    }  # Cast to string to avoid JSON serialization error
                    candidate_pool_entry["src_content"] = json.dumps(src_content)
                document_id += 1  # Increment for next entry
                outfile.write(json.dumps(candidate_pool_entry) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format CIRR images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--cirr_images_dir",
        type=str,
        default="mbeir_images/cirr_images/",
        help="Relative directory path to save CIRR images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--cirr_dir",
        type=str,
        default="src_data/cirr",
        help="Relative directory path of CIRR files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating CIRR candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting CIRR data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--split_train_into_val_and_val_into_test",
        action="store_true",
        help="Split the CIRR training set into val and move the val set to the test set.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the CIRR dataset
    # So all the paths are hardcoded.
    cirr_dir = os.path.join(args.mbeir_data_dir, args.cirr_dir)
    cirr_images_dir = os.path.join(args.mbeir_data_dir, args.cirr_images_dir)
    cirr_image_splits_dir = os.path.join(cirr_dir, "image_splits")
    cirr_candidate_pool_path = os.path.join(cirr_dir, "mbeir_cirr_cand_pool.jsonl")
    cirr_captions_dir = os.path.join(cirr_dir, "captions")

    if args.enable_image_processing:
        print(f"Processing images in {cirr_images_dir}...")
        parallel_process_image_directory(cirr_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating CIRR candidate pool in mbeir format...")
        generate_cirr_candidate_pool_from_splits(
            cirr_image_splits_dir,
            cirr_candidate_pool_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"Candidate pool saved to {cirr_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(cirr_candidate_pool_path)

    # Convert CIRR data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting CIRR data to MBEIR format...")
        data_set_list = [
            ("train", "cap.rc2.train.json"),
            ("val", "cap.rc2.val.json"),
            # ("test", "cap.rc2.test1.json"),
        ]
        for split, data_path in data_set_list:
            mbeir_format_cirr_data_path = os.path.join(cirr_dir, f"mbeir_cirr_{split}.jsonl")
            cirr_data_path = os.path.join(cirr_captions_dir, data_path)
            with open(cirr_data_path, "r") as file:
                cirr_data = json.load(file)
            mbeir_entries = cirr_to_mbeir(
                cirr_data,
                cirr_candidate_pool_path,
                args.mbeir_data_dir,
                include_src_content=True,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{CIRR_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_cirr_data_path)

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_cirr_data_path)
            print(f"MBEIR format CIRR {split} data saved to {mbeir_format_cirr_data_path}")
            print(f"Total number of entries in {mbeir_format_cirr_data_path}: {total_entries}")
            cirr_cand_pool = load_mbeir_format_pool_file_as_dict(
                cirr_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, cirr_cand_pool)

    # Split training set into val and move the val set to the test set
    if args.split_train_into_val_and_val_into_test:
        print("Split the CIRR training set into val and move the val set to the test set...")
        print("2000 of the CIRR training set will be moved to the val set.")
        mbeir_cirr_train_data_path = os.path.join(cirr_dir, "mbeir_cirr_train.jsonl")
        mbeir_cirr_train_data = load_jsonl_as_list(mbeir_cirr_train_data_path)
        random.seed(2023)
        random.shuffle(mbeir_cirr_train_data)
        cirr_new_val_data = mbeir_cirr_train_data[:2000]
        cirr_new_train_data = mbeir_cirr_train_data[2000:]
        mbeir_cirr_new_val_data_path = os.path.join(cirr_dir, "mbeir_cirr_new_val.jsonl")
        mbeir_cirr_new_train_data_path = os.path.join(cirr_dir, "mbeir_cirr_new_train.jsonl")
        mbeir_cirr_val_data_path = os.path.join(cirr_dir, "mbeir_cirr_val.jsonl")
        cirr_new_test_data = load_jsonl_as_list(mbeir_cirr_val_data_path)
        mbeir_cirr_new_test_data_path = os.path.join(cirr_dir, "mbeir_cirr_new_test.jsonl")

        # Load the candidate pool
        cirr_cand_pool = load_mbeir_format_pool_file_as_dict(
            cirr_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        save_list_as_jsonl(cirr_new_train_data, mbeir_cirr_new_train_data_path, mode="w")
        print(f"Saved new training data to {mbeir_cirr_new_train_data_path}")
        print_mbeir_format_dataset_stats(cirr_new_train_data, cirr_cand_pool)

        save_list_as_jsonl(cirr_new_val_data, mbeir_cirr_new_val_data_path, mode="w")
        print(f"Saved new val data to {mbeir_cirr_new_val_data_path}")
        print_mbeir_format_dataset_stats(cirr_new_val_data, cirr_cand_pool)

        save_list_as_jsonl(cirr_new_test_data, mbeir_cirr_new_test_data_path, mode="w")
        print(f"Saved new test data to {mbeir_cirr_new_test_data_path}")
        print_mbeir_format_dataset_stats(cirr_new_test_data, cirr_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        cirr_train_candidate_pool_path = os.path.join(cirr_dir, "mbeir_cirr_train_cand_pool.jsonl")
        mbeir_format_cirr_train_data_path = os.path.join(cirr_dir, f"mbeir_cirr_new_train.jsonl")
        assert os.path.exists(
            mbeir_format_cirr_train_data_path
        ), f"File {mbeir_format_cirr_train_data_path} does not exist"

        # Load the training data
        cirr_train_candidate_pool = {}
        cirr_cand_pool = load_mbeir_format_pool_file_as_dict(
            cirr_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_cirr_train_data = load_jsonl_as_list(mbeir_format_cirr_train_data_path)
        for entry in mbeir_format_cirr_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = cirr_cand_pool[did]
                if did not in cirr_train_candidate_pool:
                    cirr_train_candidate_pool[did] = cand
                else:
                    if cirr_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {cirr_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        cirr_train_candidate_pool_list = list(cirr_train_candidate_pool.values())
        cirr_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(cirr_train_candidate_pool_list, cirr_train_candidate_pool_path)
        print(f"Saved training candidate pool to {cirr_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(cirr_train_candidate_pool_path)


if __name__ == "__main__":
    main()

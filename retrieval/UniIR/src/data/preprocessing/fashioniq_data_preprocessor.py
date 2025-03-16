"""
fashioniq_data_preprocessor.py

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

import os
import json

FASIONIQ_QUERY_MODALITY = "image,text"
FASIONIQ_CANDIDATE_MODALITY = "image"
FASIONIQ_DATASET_ID = get_dataset_id("FashionIQ")
assert FASIONIQ_DATASET_ID is not None, "Unknown dataset name!"


def fashioniq_to_mbeir_entry(
    fashioniq_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
    concatenate_captions=True,
):
    """
    Convert Fashion IQ data format to MBEIR format.
    Sample MBEIR entry:
    {
    "qid": "0:2",
    "query_txt": null,
    "query_img_path": "...jpg",
    "query_modality": "image",
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
    dataset_id = get_dataset_id("FashionIQ")
    assert dataset_id is not None, "Unknown dataset name!"

    mbeir_entries = []

    def format_fashioniq_sentence(sentence):
        # Remove white space at the beginning and end
        sentence = sentence.strip()
        # Make the first character lowercase
        sentence = sentence[0].lower() + sentence[1:]
        # Remove the period at the end, if it exists
        if sentence.endswith("."):
            sentence = sentence[:-1]
        return sentence

    if concatenate_captions:
        # Join all captions with " and "
        captions = fashioniq_entry["captions"]
        filtered_captions = [format_fashioniq_sentence(caption) for caption in captions if caption]
        caption = " and ".join(filtered_captions)
        clean_caption = format_string(caption)
        if not clean_caption:
            print(f"Warning: Invalid query_txt : {caption}")
            return mbeir_entries  # query txt is missing
        captions = [clean_caption]
    else:
        captions = [format_string(caption) for caption in fashioniq_entry["captions"] if format_string(caption)]

    for clean_caption in captions:
        mbeir_entry = {
            "qid": None,
            "query_txt": clean_caption,
            "query_img_path": None,
            "query_modality": FASIONIQ_QUERY_MODALITY,
            "query_src_content": None,
            "pos_cand_list": [],
            "neg_cand_list": [],
        }
        query_img_name = fashioniq_entry["candidate"] + ".jpg"
        query_img_path = os.path.join("mbeir_images", "fashioniq_images", query_img_name)
        if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
            print(f"Warning: Invalid query_img_path : {query_img_path}")
            continue  # query image is missing
        mbeir_entry["query_img_path"] = query_img_path

        if include_src_content:
            query_src_content = {
                "candidate_img_id": fashioniq_entry["candidate"],
            }
            mbeir_entry["query_src_content"] = json.dumps(query_src_content)

        # Add positive candidate
        doc_key = fashioniq_entry["target"]
        pos_candidate = candidate_pool.get(doc_key, None)
        if not pos_candidate:
            print(f"Warning: No positive candidate for {doc_key}")
            continue  # positive candidate is missing
        mbeir_entry["pos_cand_list"].append(pos_candidate["did"])  # We only need the document ID

        mbeir_entries.append(mbeir_entry)

    return mbeir_entries


def load_mbeir_format_fashioniq_pool_file_as_dict(pool_file_path):
    """
    Load the FashionIQ candidate pool file into a dictionary.
    FashionIQ has unique candidate IDs the image name, so we can use them as keys.
    """
    pool_dict = {}
    assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."
    with open(pool_file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            src_content = json.loads(entry["src_content"])
            doc_key = src_content["img_id"]  # Use the image name as the key
            pool_dict[doc_key] = entry
    return pool_dict


def fashioniq_to_mbeir(
    fashioniq_data,
    candidate_pool_file_path,
    mbeir_data_dir,
    include_src_content=True,
    concatenate_captions=True,
):
    """
    FashionIQ dataset to MBEIR format.
    """
    mbeir_entries_merged = []

    # Load candidate pool
    candidate_pool = load_mbeir_format_fashioniq_pool_file_as_dict(candidate_pool_file_path)

    for fashioniq_entry in fashioniq_data:
        mbeir_entries = fashioniq_to_mbeir_entry(
            fashioniq_entry,
            candidate_pool,
            mbeir_data_dir,
            include_src_content,
            concatenate_captions,
        )
        if mbeir_entries:  # Skip invalid entries
            mbeir_entries_merged.extend(mbeir_entries)
    return mbeir_entries_merged


def generate_fashioniq_candidate_pool(
    fashioniq_images_dir,
    fashioniq_candidate_pool_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate Fashion IQ candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure of the fashioniq_images_dir:
    ├── fashioniq_images_dir
    │    ├── B00I0XXRJU.jpg
    │    ├── B00J2UZLNU.jpg
    │    └── ...
    """

    # Create the image_name_set by listing all files in the fashioniq_images_dir.
    image_name_set = {fname for fname in os.listdir(fashioniq_images_dir) if fname.endswith(".jpg")}

    document_id = 1  # Note: We start from 1 for document IDs

    with open(fashioniq_candidate_pool_path, "w") as outfile:
        for image_name in image_name_set:
            # Note: we always store relative paths to MBEIR data directory
            img_path_rel = os.path.join("mbeir_images", "fashioniq_images", image_name)
            img_path_abs = os.path.join(mbeir_data_dir, img_path_rel)

            # if the image is valid, add it to the candidate pool
            if is_valid_image(img_path_abs):
                candidate_pool_entry = {
                    "txt": None,
                    "img_path": img_path_rel,
                    "modality": "image",
                    "did": f"{FASIONIQ_DATASET_ID}:{document_id}",
                }
                if include_src_content:
                    src_content = {
                        "img_id": os.path.splitext(image_name)[0],
                    }  # Cast to string to avoid JSON serialization error
                    candidate_pool_entry["src_content"] = json.dumps(src_content)
                document_id += 1  # increment for next entry
                outfile.write(json.dumps(candidate_pool_entry) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format Fashion IQ images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--fashioniq_images_dir",
        type=str,
        default="mbeir_images/fashioniq_images/",
        help="Relative directory path to save Fashion IQ images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--fashioniq_dir",
        type=str,
        default="src_data/fashioniq",
        help="Relative directory path of Fashion IQ files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating Fashion IQ candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting Fashion IQ data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--split_train_into_val_and_val_into_test",
        action="store_true",
        help="Split the Fashion IQ training set into val and move the val set to the test set.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the Fashion IQ dataset
    # So all the paths are hardcoded.
    fashioniq_dir = os.path.join(args.mbeir_data_dir, args.fashioniq_dir)
    fashioniq_images_dir = os.path.join(args.mbeir_data_dir, args.fashioniq_images_dir)
    fashioniq_candidate_pool_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_cand_pool.jsonl")
    fashioniq_captions_dir = os.path.join(fashioniq_dir, "captions")

    if args.enable_image_processing:
        print(f"Processing images in {fashioniq_images_dir}...")
        parallel_process_image_directory(fashioniq_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating Fashion IQ candidate pool in mbeir format...")
        generate_fashioniq_candidate_pool(
            fashioniq_images_dir,
            fashioniq_candidate_pool_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"Candidate pool saved to {fashioniq_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(fashioniq_candidate_pool_path)

    # Convert Fashion IQ data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting Fashion IQ data to MBEIR format...")
        data_set_list = [
            (
                "train",
                [
                    "cap.dress.train.json",
                    "cap.shirt.train.json",
                    "cap.toptee.train.json",
                ],
            ),
            (
                "val",
                ["cap.dress.val.json", "cap.shirt.val.json", "cap.toptee.val.json"],
            ),
            # ("test", ["cap.dress.test.json", "cap.shirt.test.json", "cap.toptee.test.json"]),
        ]
        for split, data_paths in data_set_list:
            mbeir_format_fashioniq_data_path = os.path.join(fashioniq_dir, f"mbeir_fashioniq_{split}.jsonl")

            mbeir_entries_merged = []
            for data_path in data_paths:
                data_path = os.path.join(fashioniq_captions_dir, data_path)
                with open(data_path, "r") as file:
                    data = json.load(file)
                mbeir_entries = fashioniq_to_mbeir(
                    data,
                    fashioniq_candidate_pool_path,
                    args.mbeir_data_dir,
                    include_src_content=True,
                    concatenate_captions=True,
                )
                mbeir_entries_merged.extend(mbeir_entries)

            # Aggregate data
            mbeir_entries_merged = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries_merged)

            # Generate query ID after merging all the entries from three files
            for i, entry in enumerate(mbeir_entries_merged):
                entry.update({"qid": f"{FASIONIQ_DATASET_ID}:{i + 1}"})

            # Save to file
            save_list_as_jsonl(mbeir_entries_merged, mbeir_format_fashioniq_data_path)

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_fashioniq_data_path)
            print(f"MBEIR format FashionIQ {split} data saved to {mbeir_format_fashioniq_data_path}")
            print(f"Total number of entries in {mbeir_format_fashioniq_data_path}: {total_entries}")
            fashioniq_cand_pool = load_mbeir_format_pool_file_as_dict(
                fashioniq_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, fashioniq_cand_pool)

    # Split training set into val and move the val set to the test set
    if args.split_train_into_val_and_val_into_test:
        print("Split the Fashion IQ training set into val and move the val set to the test set...")
        print("1700 of the Fashion IQ training set will be moved to the val set.")
        mbeir_fashioniq_train_data_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_train.jsonl")
        mbeir_fashioniq_train_data = load_jsonl_as_list(mbeir_fashioniq_train_data_path)
        random.seed(2023)
        random.shuffle(mbeir_fashioniq_train_data)
        fashioniq_new_val_data = mbeir_fashioniq_train_data[:1700]
        fashioniq_new_train_data = mbeir_fashioniq_train_data[1700:]
        mbeir_fashioniq_new_val_data_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_new_val.jsonl")
        mbeir_fashioniq_new_train_data_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_new_train.jsonl")
        mbeir_fashioniq_val_data_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_val.jsonl")
        fashioniq_new_test_data = load_jsonl_as_list(mbeir_fashioniq_val_data_path)
        mbeir_fashioniq_new_test_data_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_new_test.jsonl")

        # Load the candidate pool
        fashioniq_cand_pool = load_mbeir_format_pool_file_as_dict(
            fashioniq_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        save_list_as_jsonl(fashioniq_new_train_data, mbeir_fashioniq_new_train_data_path, mode="w")
        print(f"Saved new training data to {mbeir_fashioniq_new_train_data_path}")
        print_mbeir_format_dataset_stats(fashioniq_new_train_data, fashioniq_cand_pool)

        save_list_as_jsonl(fashioniq_new_val_data, mbeir_fashioniq_new_val_data_path, mode="w")
        print(f"Saved new val data to {mbeir_fashioniq_new_val_data_path}")
        print_mbeir_format_dataset_stats(fashioniq_new_val_data, fashioniq_cand_pool)

        save_list_as_jsonl(fashioniq_new_test_data, mbeir_fashioniq_new_test_data_path, mode="w")
        print(f"Saved new test data to {mbeir_fashioniq_new_test_data_path}")
        print_mbeir_format_dataset_stats(fashioniq_new_test_data, fashioniq_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        fashioniq_train_candidate_pool_path = os.path.join(fashioniq_dir, "mbeir_fashioniq_train_cand_pool.jsonl")
        mbeir_format_fashioniq_train_data_path = os.path.join(fashioniq_dir, f"mbeir_fashioniq_new_train.jsonl")
        assert os.path.exists(
            mbeir_format_fashioniq_train_data_path
        ), f"File {mbeir_format_fashioniq_train_data_path} does not exist"

        # Load the training data
        fashioniq_train_candidate_pool = {}
        fashioniq_cand_pool = load_mbeir_format_pool_file_as_dict(
            fashioniq_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_fashioniq_train_data = load_jsonl_as_list(mbeir_format_fashioniq_train_data_path)
        for entry in mbeir_format_fashioniq_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = fashioniq_cand_pool[did]
                if did not in fashioniq_train_candidate_pool:
                    fashioniq_train_candidate_pool[did] = cand
                else:
                    if fashioniq_train_candidate_pool[did] != cand:
                        print(
                            f"Duplicate did for two candidates found: {fashioniq_train_candidate_pool[did]} and {cand}"
                        )

        # Save the training candidate pool
        fashioniq_train_candidate_pool_list = list(fashioniq_train_candidate_pool.values())
        fashioniq_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(fashioniq_train_candidate_pool_list, fashioniq_train_candidate_pool_path)
        print(f"Saved training candidate pool to {fashioniq_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(fashioniq_train_candidate_pool_path)


if __name__ == "__main__":
    main()

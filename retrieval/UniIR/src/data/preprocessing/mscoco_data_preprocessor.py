"""
mscoco_data_preprocessor.py

Module description:
    1. Convert all image files to the JPG format and resize the smaller dimension to 256.
    2. Generate candidate pool.
    3. Convert the dataset to the MBEIR format.
"""

import os
import zipfile
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

MSCOCO_QUERY_MODALITY_IMAGE = "image"
MSCOCO_QUERY_MODALITY_TEXT = "text"
MSCOCO_CANDIDATE_MODALITY_IMAGE = "image"
MSCOCO_CANDIDATE_MODALITY_TEXT = "text"
MSCOCO_DATASET_ID = get_dataset_id("MSCOCO")
assert MSCOCO_DATASET_ID is not None, "Unknown dataset name!"


def mscoco_to_mbeir_entry(
    mscoco_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert MSCOCO data format to MBEIR format.
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
    mbeir_entries = []
    img_path = mscoco_entry["image"]
    # Note: we always store relative paths to MBEIR data directory
    sub_directory, base_filename_with_ext = os.path.split(img_path)  # e.g. "train2014", "COCO_train2014_009.jpg"
    base_filename = os.path.splitext(base_filename_with_ext)[0]
    img_path = os.path.join("mbeir_images", "mscoco_images", sub_directory, base_filename + ".jpg")
    if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
        print(f"Warning: Invalid image: {img_path}")  # if the image is invalid, skip it
        return None

    # val.json has a list of captions, while train.json has a single caption
    captions = mscoco_entry["caption"] if isinstance(mscoco_entry["caption"], list) else [mscoco_entry["caption"]]

    # Generate image to text MBEIR entry
    mbeir_entry_img2txt = {
        "qid": None,
        "query_txt": None,
        "query_img_path": img_path,
        "query_modality": MSCOCO_QUERY_MODALITY_IMAGE,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    for caption in captions[:5]:  # Only use the first 5 captions
        txt = format_string(caption)
        if not txt:
            print(f"Warning: Empty caption: {mscoco_entry}")
            continue

        # Add positive candidate to img2txt entry
        _img2txt_candidate = {
            "txt": txt,
            "modality": MSCOCO_CANDIDATE_MODALITY_TEXT,
        }
        doc_key = generate_mbeir_format_doc_key(_img2txt_candidate)
        img2txt_candidate = candidate_pool.get(doc_key, None)
        assert img2txt_candidate, f"Cannot find candidate for {doc_key}"
        mbeir_entry_img2txt["pos_cand_list"].append(img2txt_candidate["did"])  # We only store the document ID

        # Generate text to image MBEIR entry
        mbeir_entry_txt2img = {
            "qid": None,
            "query_txt": txt,
            "query_img_path": None,
            "query_modality": MSCOCO_QUERY_MODALITY_TEXT,
            "query_src_content": None,
            "pos_cand_list": [],
            "neg_cand_list": [],
        }

        # Add positive candidates to txt2img entry
        _txt2img_candidate = {
            "img_path": img_path,
            "modality": MSCOCO_CANDIDATE_MODALITY_IMAGE,
        }
        doc_key = generate_mbeir_format_doc_key(_txt2img_candidate)
        txt2img_candidate = candidate_pool.get(doc_key, None)
        assert txt2img_candidate, f"Cannot find candidate for {doc_key}"
        mbeir_entry_txt2img["pos_cand_list"].append(txt2img_candidate["did"])

        mbeir_entries.append(mbeir_entry_txt2img)
    mbeir_entries.append(mbeir_entry_img2txt)
    assert mbeir_entries, f"Cannot find positive image facts for {captions}"
    return mbeir_entries


def mscoco_to_mbeir(mscoco_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    """
    mscoco dataset to MBEIR format.
    """
    mbeir_entries_merged = []

    # Load candidate pool
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(candidate_pool_file_path, doc_key_to_content=True)

    for mscoco_entry in mscoco_data:
        mbeir_entries = mscoco_to_mbeir_entry(
            mscoco_entry,
            cand_pool_dict,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entries:  # Skip invalid entries
            mbeir_entries_merged.extend(mbeir_entries)
    return mbeir_entries_merged


def generate_mscoco_candidate_pool(
    mscoco_dir,
    mbeir_data_dir,
    mscoco_candidate_pool_path,
    mscoco_txt_val_candidate_pool_path,
    mscoco_txt_test_candidate_pool_path,
    mscoco_img_val_candidate_pool_path,
    mscoco_img_test_candidate_pool_path,
    include_src_content=True,
):
    """
    Generate mscoco candidate pool in mbeir format and save it to a jsonl file.
    Here is the expected directory structure of the mscoco_images_dir:
    mscoco_dir/
        ├── coco_karpathy_test.json
        ├── coco_karpathy_train.json
        ├── coco_karpathy_val.json
    """

    mscoco_data_files = [
        os.path.join(mscoco_dir, filename)
        for filename in os.listdir(mscoco_dir)
        if filename.endswith(".json") and "coco_karpathy" in filename  # look for the mentioned json files
    ]

    document_id = 1  # Note: We start from 1 for document IDs
    seen_txts_all = {}  # To store description that we've already seen
    seen_image_paths_all = {}  # To store image paths that we've already seen

    # Separate seen sets for validation and test data
    seen_txts_val = set()
    seen_image_paths_val = set()
    seen_txts_test = set()
    seen_image_paths_test = set()

    def write_to_file(file, data):
        file.write(json.dumps(data) + "\n")

    # Open separate files for text and image entries
    with (
        open(mscoco_candidate_pool_path, "w") as allfile,
        open(mscoco_txt_val_candidate_pool_path, "w") as txt_val_file,
        open(mscoco_txt_test_candidate_pool_path, "w") as txt_test_file,
        open(mscoco_img_val_candidate_pool_path, "w") as img_val_file,
        open(mscoco_img_test_candidate_pool_path, "w") as img_test_file,
    ):
        for mscoco_data_file in mscoco_data_files:
            is_val = "val" in os.path.basename(mscoco_data_file)
            is_test = "test" in os.path.basename(mscoco_data_file)

            with open(mscoco_data_file, "r") as source:
                mscoco_data = json.load(source)

                for mscoco_entry in mscoco_data:
                    img_path = mscoco_entry["image"]
                    # Note: we always store relative paths to MBEIR data directory
                    sub_directory, base_filename_with_ext = os.path.split(
                        img_path
                    )  # e.g. "train2014", "COCO_train2014_009.jpg"
                    base_filename = os.path.splitext(base_filename_with_ext)[0]
                    img_path = os.path.join(
                        "mbeir_images",
                        "mscoco_images",
                        sub_directory,
                        base_filename + ".jpg",
                    )
                    if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                        print(f"Warning: Invalid image: {img_path}")
                    else:
                        candidate_pool_entry_img = {
                            "txt": None,
                            "img_path": img_path,
                            "modality": MSCOCO_CANDIDATE_MODALITY_IMAGE,
                            "did": f"{MSCOCO_DATASET_ID}:{document_id}",
                            "src_content": None,
                        }
                        # If image path hasn't been seen, create image entry
                        if img_path not in seen_image_paths_all:
                            write_to_file(allfile, candidate_pool_entry_img)
                            seen_image_paths_all[img_path] = candidate_pool_entry_img
                            document_id += 1  # increment for next entry
                        else:
                            candidate_pool_entry_img = seen_image_paths_all[img_path]
                        if is_val and img_path not in seen_image_paths_val:
                            write_to_file(img_val_file, candidate_pool_entry_img)
                            seen_image_paths_val.add(img_path)
                        elif is_test and img_path not in seen_image_paths_test:
                            write_to_file(img_test_file, candidate_pool_entry_img)
                            seen_image_paths_test.add(img_path)

                    captions = (
                        mscoco_entry["caption"]
                        if isinstance(mscoco_entry["caption"], list)
                        else [mscoco_entry["caption"]]
                    )
                    for caption in captions[:5]:  # Only use the first 5 captions
                        txt = format_string(caption)
                        if not txt:
                            print(f"Warning: Empty caption: {mscoco_entry}")  # if the caption is empty, skip it
                            continue

                        candidate_pool_entry_txt = {
                            "txt": txt,
                            "img_path": None,
                            "modality": MSCOCO_CANDIDATE_MODALITY_TEXT,
                            "did": f"{MSCOCO_DATASET_ID}:{document_id}",
                            "src_content": None,
                        }
                        # If description hasn't been seen, create text entry
                        if txt not in seen_txts_all:
                            write_to_file(allfile, candidate_pool_entry_txt)
                            seen_txts_all[txt] = candidate_pool_entry_txt
                            document_id += 1  # increment for next entry
                        else:
                            candidate_pool_entry_txt = seen_txts_all[txt]
                        if is_val and txt not in seen_txts_val:
                            write_to_file(txt_val_file, candidate_pool_entry_txt)
                            seen_txts_val.add(txt)
                        elif is_test and txt not in seen_txts_test:
                            write_to_file(txt_test_file, candidate_pool_entry_txt)
                            seen_txts_test.add(txt)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format mscoco images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--mscoco_images_dir",
        type=str,
        default="mbeir_images/mscoco_images/",
        help="Relative directory path to save mscoco images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--mscoco_dir",
        type=str,
        default="src_data/mscoco",
        help="Relative directory path of mscoco files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating mscoco candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting mscoco data to MBEIR format.",
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
        "--separate_val_test_to_txt_img",
        action="store_true",
        help="Separate the val and test data into txt and img files.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download and unzip MSCOCO image files the JSON files.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the mscoco dataset
    # So all the paths are hardcoded.
    mscoco_dir = os.path.join(args.mbeir_data_dir, args.mscoco_dir)
    mscoco_images_dir = os.path.join(args.mbeir_data_dir, args.mscoco_images_dir)
    mscoco_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_cand_pool.jsonl")

    if args.download:
        json_urls = {
            "train": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_train.json",
            "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val.json",
            "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test.json",
        }
        zip_urls = {
            "train2014": "http://images.cocodataset.org/zips/train2014.zip",
            "val2014": "http://images.cocodataset.org/zips/val2014.zip",
        }

        total_entries = 0
        total_captions = 0

        def download_file(url, filename):
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(filename, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

        def unzip_file(zip_filepath, dest_dir):
            with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
                print(f"Unzipped {zip_filepath} to {dest_dir} successfully!")

        mscoco_data_dir = os.path.join(args.mbeir_data_dir, args.mscoco_data_dir)
        for key, url in json_urls.items():
            coco_data_path = os.path.join(mscoco_data_dir, f"coco_karpathy_{key}.json")
            download_file(url, coco_data_path)
            print(f"Downloaded {coco_data_path} successfully!")

            def count_entries_and_captions(filename):
                with open(filename, "r") as file:
                    data = json.load(file)
                    # If the file is train.json, every 5 entries correspond to one image.
                    if "train" in filename:
                        num_entries = len(data) // 5
                        num_captions = len(data)
                    else:
                        num_entries = len(data)
                        num_captions = len(data) * 5
                    return num_entries, num_captions

            num_entries, num_captions = count_entries_and_captions(coco_data_path)
            print(f"{coco_data_path} - Number of images: {num_entries}, Number of captions: {num_captions}")
            total_entries += num_entries
            total_captions += num_captions

        print(f"Total number of entries (images): {total_entries}")
        print(f"Total number of captions: {total_captions}")

        mscoco_images_dir = os.path.join(args.mbeir_data_dir, args.mscoco_images_dir)
        os.makedirs(mscoco_images_dir, exist_ok=True)
        for key, url in zip_urls.items():
            coco_images_path = os.path.join(mscoco_images_dir, f"{key}.zip")
            download_file(url, coco_images_path)
            print(f"Downloaded {coco_images_path} successfully!")
            unzip_file(coco_images_path, mscoco_images_dir)

    if args.enable_image_processing:
        print(f"Processing images in {mscoco_images_dir}...")
        parallel_process_image_directory(mscoco_images_dir, num_processes=cpu_count())

    # Generate candidate pool
    if args.enable_candidate_pool:
        mscoco_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_cand_pool.jsonl")
        mscoco_txt_val_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_txt_val_cand_pool.jsonl")
        mscoco_txt_test_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_txt_test_cand_pool.jsonl")
        mscoco_img_val_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_img_val_cand_pool.jsonl")
        mscoco_img_test_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_img_test_cand_pool.jsonl")
        print("Generating mscoco candidate pool in mbeir format...")
        generate_mscoco_candidate_pool(
            mscoco_dir,
            args.mbeir_data_dir,
            mscoco_candidate_pool_path,
            mscoco_txt_val_candidate_pool_path,
            mscoco_txt_test_candidate_pool_path,
            mscoco_img_val_candidate_pool_path,
            mscoco_img_test_candidate_pool_path,
        )
        print(f"Candidate pool saved to {mscoco_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_candidate_pool_path)
        print(f"Validation text candidate pool saved to {mscoco_txt_val_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_txt_val_candidate_pool_path)
        print(f"Test text candidate pool saved to {mscoco_txt_test_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_txt_test_candidate_pool_path)
        print(f"Validation image candidate pool saved to {mscoco_img_val_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_img_val_candidate_pool_path)
        print(f"Test image candidate pool saved to {mscoco_img_test_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_img_test_candidate_pool_path)

    # Convert mscoco data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting mscoco data to MBEIR format...")

        data_splits_paths = [
            ("train", os.path.join(mscoco_dir, "coco_karpathy_train.json")),
            ("val", os.path.join(mscoco_dir, "coco_karpathy_val.json")),
            ("test", os.path.join(mscoco_dir, "coco_karpathy_test.json")),
        ]
        for split, data_path in data_splits_paths:
            mbeir_format_mscoco_file_path = os.path.join(mscoco_dir, f"mbeir_mscoco_{split}.jsonl")
            with open(data_path, "r") as source:
                mscoco_data = json.load(source)
            mbeir_entries = mscoco_to_mbeir(
                mscoco_data,
                mscoco_candidate_pool_path,
                args.mbeir_data_dir,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries, print_duplicate=False)

            # Trim the training data queries for text2image
            if split == "train":
                txt2img_entries = []
                img2txt_entries = []
                for entry in mbeir_entries:
                    if entry["query_modality"] == "text":
                        txt2img_entries.append(entry)
                    else:
                        img2txt_entries.append(entry)
                random.seed(2023)
                random.shuffle(txt2img_entries)
                mbeir_entries = txt2img_entries[:100000] + img2txt_entries
                random.shuffle(mbeir_entries)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{MSCOCO_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_mscoco_file_path, mode="w")

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_mscoco_file_path)
            print(f"MBEIR format MMSCOCO {split} data saved to {mbeir_format_mscoco_file_path}")
            print(f"Total number of entries in {mbeir_format_mscoco_file_path}: {total_entries}")
            mscoco_cand_pool = load_mbeir_format_pool_file_as_dict(
                mscoco_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, mscoco_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        mscoco_train_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_train_cand_pool.jsonl")
        mbeir_format_mscoco_train_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_train.jsonl")
        assert os.path.exists(
            mbeir_format_mscoco_train_data_path
        ), f"File {mbeir_format_mscoco_train_data_path} does not exist"

        # Load the training data
        mscoco_train_candidate_pool = {}
        mscoco_cand_pool = load_mbeir_format_pool_file_as_dict(
            mscoco_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_mscoco_train_data = load_jsonl_as_list(mbeir_format_mscoco_train_data_path)
        for entry in mbeir_format_mscoco_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = mscoco_cand_pool[did]
                if did not in mscoco_train_candidate_pool:
                    mscoco_train_candidate_pool[did] = cand
                else:
                    if mscoco_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {mscoco_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        mscoco_train_candidate_pool_list = list(mscoco_train_candidate_pool.values())
        mscoco_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(mscoco_train_candidate_pool_list, mscoco_train_candidate_pool_path)
        print(f"Saved training candidate pool to {mscoco_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(mscoco_train_candidate_pool_path)

    # Separate the val and test data into txt and img files
    if args.separate_val_test_to_txt_img:
        print("Separating val and test data into txt and img files...")
        mbeir_format_mscoco_val_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_val.jsonl")
        mbeir_format_mscoco_test_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_test.jsonl")
        assert os.path.exists(
            mbeir_format_mscoco_val_data_path
        ), f"File {mbeir_format_mscoco_val_data_path} does not exist"
        assert os.path.exists(
            mbeir_format_mscoco_test_data_path
        ), f"File {mbeir_format_mscoco_test_data_path} does not exist"

        mbeir_format_mscoco_txt_val_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_txt_val.jsonl")
        mbeir_format_mscoco_txt_test_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_txt_test.jsonl")
        mbeir_format_mscoco_img_val_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_img_val.jsonl")
        mbeir_format_mscoco_img_test_data_path = os.path.join(mscoco_dir, f"mbeir_mscoco_img_test.jsonl")

        mbeir_format_mscoco_val_data = load_jsonl_as_list(mbeir_format_mscoco_val_data_path)
        mbeir_format_mscoco_test_data = load_jsonl_as_list(mbeir_format_mscoco_test_data_path)

        mbeir_format_mscoco_txt_val_data = []
        mbeir_format_mscoco_txt_test_data = []
        mbeir_format_mscoco_img_val_data = []
        mbeir_format_mscoco_img_test_data = []

        for entry in mbeir_format_mscoco_val_data:
            if entry["query_modality"] == "text":
                mbeir_format_mscoco_txt_val_data.append(entry)
            else:
                mbeir_format_mscoco_img_val_data.append(entry)

        for entry in mbeir_format_mscoco_test_data:
            if entry["query_modality"] == "text":
                mbeir_format_mscoco_txt_test_data.append(entry)
            else:
                mbeir_format_mscoco_img_test_data.append(entry)

        save_list_as_jsonl(mbeir_format_mscoco_txt_val_data, mbeir_format_mscoco_txt_val_data_path)
        save_list_as_jsonl(mbeir_format_mscoco_txt_test_data, mbeir_format_mscoco_txt_test_data_path)
        save_list_as_jsonl(mbeir_format_mscoco_img_val_data, mbeir_format_mscoco_img_val_data_path)
        save_list_as_jsonl(mbeir_format_mscoco_img_test_data, mbeir_format_mscoco_img_test_data_path)

        print(f"Saved val text data to {mbeir_format_mscoco_txt_val_data_path}")
        print(f"Saved test text data to {mbeir_format_mscoco_txt_test_data_path}")
        print(f"Saved val image data to {mbeir_format_mscoco_img_val_data_path}")
        print(f"Saved test image data to {mbeir_format_mscoco_img_test_data_path}")

        # Print statistics
        mscoco_txt_val_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_txt_val_cand_pool.jsonl")
        mscoco_txt_test_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_txt_test_cand_pool.jsonl")
        mscoco_img_val_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_img_val_cand_pool.jsonl")
        mscoco_img_test_candidate_pool_path = os.path.join(mscoco_dir, "mbeir_mscoco_img_test_cand_pool.jsonl")
        total_entries, _data = count_entries_in_file(mbeir_format_mscoco_txt_val_data_path)
        print(f"Total number of entries in {mbeir_format_mscoco_txt_val_data_path}: {total_entries}")
        mscoco_img_val_candidate_pool = load_mbeir_format_pool_file_as_dict(
            mscoco_img_val_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(_data, mscoco_img_val_candidate_pool)
        total_entries, _data = count_entries_in_file(mbeir_format_mscoco_txt_test_data_path)
        print(f"Total number of entries in {mbeir_format_mscoco_txt_test_data_path}: {total_entries}")
        mscoco_img_test_candidate_pool = load_mbeir_format_pool_file_as_dict(
            mscoco_img_test_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(_data, mscoco_img_test_candidate_pool)
        total_entries, _data = count_entries_in_file(mbeir_format_mscoco_img_val_data_path)
        print(f"Total number of entries in {mbeir_format_mscoco_img_val_data_path}: {total_entries}")
        mscoco_txt_val_candidate_pool = load_mbeir_format_pool_file_as_dict(
            mscoco_txt_val_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(_data, mscoco_txt_val_candidate_pool)
        total_entries, _data = count_entries_in_file(mbeir_format_mscoco_img_test_data_path)
        print(f"Total number of entries in {mbeir_format_mscoco_img_test_data_path}: {total_entries}")
        mscoco_txt_test_candidate_pool = load_mbeir_format_pool_file_as_dict(
            mscoco_txt_test_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        print_mbeir_format_dataset_stats(_data, mscoco_txt_test_candidate_pool)


if __name__ == "__main__":
    main()

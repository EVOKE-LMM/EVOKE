import json
import argparse
import os
import random
from PIL import Image

from multiprocessing import Pool, cpu_count

from utils import (
    format_string,
    get_dataset_id,
    is_valid_image,
    load_jsonl_as_list,
    count_entries_in_file,
    check_duplicates,
    load_mbeir_format_pool_file_as_dict,
    save_list_as_jsonl,
    print_mbeir_format_dataset_stats,
    print_mbeir_format_cand_pool_stats,
    parallel_process_image_directory,
    aggregate_candidates_for_mbeir_format_dataset,
    generate_mbeir_format_doc_key,
)

VISUALNEWS_QUERY_MODALITY_IMAGE = "image"
VISUALNEWS_QUERY_MODALITY_TEXT = "text"
VISUALNEWS_CANDIDATE_MODALITY_IMAGE = "image"
VISUALNEWS_CANDIDATE_MODALITY_TEXT = "text"
VISUALNEWS_DATASET_ID = get_dataset_id("VisualNews")
assert VISUALNEWS_DATASET_ID is not None, "Unknown dataset name!"


def visualnews_to_mbeir_entry(
    visualnews_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=False,
):
    """
    Convert Visual News entry format to MBEIR format.
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

    # Note: we always store relative paths to MBEIR data directory
    # Here we remove "./" from the image path e.g. './guardian/images/.../...jpg'
    img_path = os.path.join("mbeir_images", "visualnews_images", visualnews_entry["image_path"][2:])
    if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
        print(f"Warning: Invalid image: {img_path}")  # if the image is invalid, skip it
        return None

    txt = format_string(visualnews_entry["caption"])
    if not txt:
        print(f"Warning: Empty caption: {visualnews_entry}")
        return None

    # Generate image to text task data
    mbeir_entry_img2txt = {
        "qid": None,
        "query_txt": None,
        "query_img_path": img_path,
        "query_modality": VISUALNEWS_QUERY_MODALITY_IMAGE,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }
    _img2txt_candidate = {
        "txt": txt,
        "modality": VISUALNEWS_CANDIDATE_MODALITY_TEXT,
    }
    doc_key = generate_mbeir_format_doc_key(_img2txt_candidate)
    img2txt_candidate_did = candidate_pool.get(doc_key, None)
    assert img2txt_candidate_did, f"Cannot find candidate for {doc_key}"
    mbeir_entry_img2txt["pos_cand_list"].append(img2txt_candidate_did)
    mbeir_entries.append(mbeir_entry_img2txt)

    # Generate text to image task data
    mbeir_entry_txt2img = {
        "qid": None,
        "query_txt": txt,
        "query_img_path": None,
        "query_modality": VISUALNEWS_QUERY_MODALITY_TEXT,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }
    # Add positive candidates to txt2img entry
    _txt2img_candidate = {
        "img_path": img_path,
        "modality": VISUALNEWS_CANDIDATE_MODALITY_IMAGE,
    }
    doc_key = generate_mbeir_format_doc_key(_txt2img_candidate)
    txt2img_candidate_did = candidate_pool.get(doc_key, None)
    assert txt2img_candidate_did, f"Cannot find candidate for {doc_key}"
    mbeir_entry_txt2img["pos_cand_list"].append(txt2img_candidate_did)
    mbeir_entries.append(mbeir_entry_txt2img)

    assert mbeir_entries, "MBEIR entries cannot be empty."
    return mbeir_entries


def visualnews_to_mbeir(
    visualnews_data,
    candidate_pool_file_path,
    mbeir_data_dir,
    include_src_content=False,
):
    mbeir_entries_merged = []

    # Load candidate pool
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(candidate_pool_file_path, doc_key_to_content=False)

    for visualnews_entry in visualnews_data:
        mbeir_entries = visualnews_to_mbeir_entry(
            visualnews_entry,
            cand_pool_dict,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entries:  # Skip invalid entries
            mbeir_entries_merged.extend(mbeir_entries)
    return mbeir_entries_merged


def generate_visualnews_candidate_pool(
    visualnews_source_file_path,
    visualnews_cand_pool_path,
    mbeir_data_dir,
):
    """
    Generate Visual News candidate pool in mbeir format.
    """

    with open(visualnews_source_file_path, "r") as source:
        visualnews_data = json.load(source)
        document_id = 1  # Note: We start from 1 for document IDs
        seen_txts = set()  # To store captions that we've already seen
        seen_image_paths = set()  # To store image paths that we've already seen

        with open(visualnews_cand_pool_path, "w") as outfile:
            for visualnews_entry in visualnews_data:
                # Note: we always store relative paths to MBEIR data directory
                # Here we remove "./" from the image path e.g. './guardian/images/.../...jpg'
                img_path = os.path.join(
                    "mbeir_images",
                    "visualnews_images",
                    visualnews_entry["image_path"][2:],
                )
                caption = format_string(visualnews_entry["caption"])  # Capitalize the first letter of each sentence

                # Track if we've seen both the caption and image path
                seen_both = caption in seen_txts and img_path in seen_image_paths

                if not seen_both:
                    # If caption hasn't been seen, create text entry
                    if caption not in seen_txts:
                        # If the description is empty, skip it
                        if not caption:
                            print(f"Warning: Empty caption: {img_path}")
                        else:
                            candidate_pool_entry_txt = {
                                "txt": caption,
                                "img_path": None,
                                "modality": "text",
                                "did": f"{VISUALNEWS_DATASET_ID}:{document_id}",
                            }
                            document_id += 1  # increment for next entry
                            outfile.write(json.dumps(candidate_pool_entry_txt) + "\n")
                            seen_txts.add(caption)

                    # If image path hasn't been seen, create image entry
                    if img_path not in seen_image_paths:
                        # If image is not valid, skip it
                        if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                            print(f"Warning: Invalid image: {img_path}")
                        else:
                            candidate_pool_entry_img = {
                                "txt": None,
                                "img_path": img_path,
                                "modality": "image",
                                "did": f"{VISUALNEWS_DATASET_ID}:{document_id}",
                            }
                            document_id += 1  # increment for next entry
                            outfile.write(json.dumps(candidate_pool_entry_img) + "\n")
                            seen_image_paths.add(img_path)


def split_data(data, train_samples, val_samples, test_samples):
    sources = ["washington_post", "guardian", "bbc", "usa_today"]

    train_data, val_data, test_data = [], [], []

    for source in sources:
        # Get all samples for the current source
        source_data = [entry for entry in data if entry["source"] == source]
        # Shuffle the data for randomness
        random.seed(2023)
        random.shuffle(source_data)

        train_data.extend(source_data[:train_samples])
        val_data.extend(source_data[train_samples : train_samples + val_samples])
        test_data.extend(source_data[train_samples + val_samples : train_samples + val_samples + test_samples])

    return train_data, val_data, test_data


def count_empty_captions(data):
    """Count and return entries with empty captions."""
    return [entry for entry in data if not entry["caption"].strip()]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process visualnews data.json file.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--visualnews_images_dir",
        type=str,
        default="mbeir_images/visualnews_images/",
        help="Relative directory path to save VisualNews images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--visualnews_dir",
        type=str,
        default="src_data/visualnews/",
        help="Relative directory path of VisualNews files under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="Resize and convert all images in the given directory to JPG format.",
    )
    parser.add_argument(
        "--enable_text_processing",
        action="store_true",
        help="Remove entries with empty captions from the data file.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting VisualNews data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_data_split",
        action="store_true",
        help="Enable splitting the data into train, val, and test sets.",
    )
    parser.add_argument(
        "--enable_all",
        action="store_true",
        help="Enable all processing steps: image processing, text processing, candidate pool generation, ",
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=25000,
        help="Number of samples for each news source in the training set.",
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=5000,
        help="Number of samples for each news source in the validation set.",
    )
    parser.add_argument(
        "--test_samples",
        type=int,
        default=5000,
        help="Number of samples for each news source in the test set.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--trim_candidate_pool",
        action="store_true",
        help="Enable trimming the number of candidates in the candidate pool.",
    )
    parser.add_argument(
        "--split_candidate_pool_by_task",
        action="store_true",
        help="Enable splitting the candidate pool according to task.",
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
    # Note: we keep the original project structure as in the VisualNews dataset
    # So all the paths are hardcoded.
    visualnews_dir = os.path.join(args.mbeir_data_dir, args.visualnews_dir)
    visualnews_data_file_path = os.path.join(visualnews_dir, "origin", "data.json")
    visualnews_images_dir = os.path.join(args.mbeir_data_dir, args.visualnews_images_dir)
    visualnews_candidate_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_cand_pool.jsonl")
    visualnews_1m_cand_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_1m_cand_pool.jsonl")

    # Process text(remove empty captions) This Could be removed.
    if args.enable_text_processing:
        print(f"Processing text in {visualnews_data_file_path}...")
        # Print statistics
        total_entries, data = count_entries_in_file(visualnews_data_file_path)
        empty_captions = count_empty_captions(data)
        print(f"Total number of entries in {visualnews_data_file_path}: {total_entries}")
        print(f"Total number of empty captions: {len(empty_captions)}")

        # Remove entries with empty captions
        cleaned_data = [entry for entry in data if entry["caption"].strip()]

        # Save cleaned data
        with open(visualnews_data_file_path, "w") as file:
            json.dump(cleaned_data, file, indent=4)
        print(f"Cleaned data saved to {visualnews_data_file_path}")

        # Re-load cleaned data
        total_entries, data = count_entries_in_file(visualnews_data_file_path)

        # Check for duplicates in cleaned data
        duplicate_images_cleaned = check_duplicates(data, "image_path")
        duplicate_articles_cleaned = check_duplicates(data, "article_path")
        duplicate_captions_cleaned = check_duplicates(data, "caption")

        # Print cleaned data statistics
        print(f"Total number of entries in {visualnews_data_file_path}: {total_entries}")
        print(f"Number of duplicate image paths in cleaned data: {len(duplicate_images_cleaned)}")
        print(f"Number of duplicate article paths in cleaned data: {len(duplicate_articles_cleaned)}")
        print(f"Number of duplicate captions in cleaned data: {len(duplicate_captions_cleaned)}")

    # Process images
    if args.enable_image_processing:
        print(f"Processing images in {visualnews_images_dir}...")
        parallel_process_image_directory(visualnews_images_dir, num_processes=cpu_count())

    # Generate candidate pool in MBEIR format
    if args.enable_candidate_pool:
        print(f"Generating VisualNews candidate pool in MBEIR format...")
        generate_visualnews_candidate_pool(
            visualnews_data_file_path,
            visualnews_candidate_pool_path,
            args.mbeir_data_dir,
        )
        print(f"Candidate pool saved to {visualnews_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(visualnews_candidate_pool_path)

    # Split clean data into train, val, and test sets
    if args.enable_data_split:
        print(f"Splitting data into train, val, and test sets...")
        _, data = count_entries_in_file(visualnews_data_file_path)
        train_data, val_data, test_data = split_data(data, args.train_samples, args.val_samples, args.test_samples)
        assert len(train_data) == args.train_samples * 4  # 4 news sources
        assert len(val_data) == args.val_samples * 4
        assert len(test_data) == args.test_samples * 4

        save_list_as_jsonl(train_data, os.path.join(visualnews_dir, "train.jsonl"))
        save_list_as_jsonl(val_data, os.path.join(visualnews_dir, "val.jsonl"))
        save_list_as_jsonl(test_data, os.path.join(visualnews_dir, "test.jsonl"))

        # Print statistics
        total_entries, _ = count_entries_in_file(os.path.join(visualnews_dir, "train.jsonl"))
        print(f"Total number of entries in {os.path.join(visualnews_dir, 'train.jsonl')}: {total_entries}")
        total_entries, _ = count_entries_in_file(os.path.join(visualnews_dir, "val.jsonl"))
        print(f"Total number of entries in {os.path.join(visualnews_dir, 'val.jsonl')}: {total_entries}")
        total_entries, _ = count_entries_in_file(os.path.join(visualnews_dir, "test.jsonl"))
        print(f"Total number of entries in {os.path.join(visualnews_dir, 'test.jsonl')}: {total_entries}")

    # Convert VisualNews data to MBEIR format
    if args.enable_mbeir_conversion:
        print(f"Converting split VisualNews data to MBEIR format.")
        data_split_list = ["train", "val", "test"]
        for data_split in data_split_list:
            visualnews_data_file_path = os.path.join(visualnews_dir, f"{data_split}.jsonl")
            mbeir_format_visualnews_path = os.path.join(visualnews_dir, f"mbeir_visualnews_{data_split}.jsonl")
            visualnews_data = load_jsonl_as_list(visualnews_data_file_path)
            mbeir_entries = visualnews_to_mbeir(
                visualnews_data,
                visualnews_candidate_pool_path,
                args.mbeir_data_dir,
                include_src_content=False,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{VISUALNEWS_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_visualnews_path)

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_visualnews_path)
            print(f"MBEIR format VisualNews data saved to {mbeir_format_visualnews_path}")
            print(f"Total number of entries in {mbeir_format_visualnews_path}: {total_entries}")
            visualnews_cand_pool = load_mbeir_format_pool_file_as_dict(
                visualnews_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, visualnews_cand_pool)

    # Trim the number of candidates in the candidate pool
    if args.trim_candidate_pool:
        print("Trim 2.5M candidate pool to 1M candidates")
        visualnews_data = []
        for split in ["train", "val", "test"]:
            data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_{split}.jsonl")
            visualnews_data.extend(load_jsonl_as_list(data_path))

        skip_pool_set = set()
        for visualnews_entry in visualnews_data:
            for cand_did in visualnews_entry["pos_cand_list"]:
                skip_pool_set.add(cand_did)
        print(f"{len(skip_pool_set)} candidates to skip")

        # Load the 2.5M candidate pool
        visualnews_cand_pool = load_jsonl_as_list(visualnews_candidate_pool_path)
        visualnews_cand_pool_without_skip_set = []
        visualnews_cand_pool_skip_set = []
        for entry in visualnews_cand_pool:
            cand_did = entry["did"]
            if cand_did not in skip_pool_set:
                visualnews_cand_pool_without_skip_set.append(entry)
            else:
                visualnews_cand_pool_skip_set.append(entry)

        # Random sample 1M candidates
        random.shuffle(visualnews_cand_pool_without_skip_set)
        augment_size = 800000
        print(f"Sample {augment_size} candidates from {len(visualnews_cand_pool_without_skip_set)} candidates")
        visualnews_cand_pool_without_skip_set = visualnews_cand_pool_without_skip_set[:augment_size]

        # Reassign document ids
        visualnews_1m_cand_pool = visualnews_cand_pool_skip_set + visualnews_cand_pool_without_skip_set
        document_id_start = 1
        oldid_newdid_map = {}
        for i, entry in enumerate(visualnews_1m_cand_pool):
            olddid = entry["did"]
            assert olddid not in oldid_newdid_map
            entry["did"] = f"{VISUALNEWS_DATASET_ID}:{document_id_start + i}"
            newdid = entry["did"]
            oldid_newdid_map[olddid] = newdid
        save_list_as_jsonl(visualnews_1m_cand_pool, visualnews_1m_cand_pool_path, mode="w")
        print_mbeir_format_cand_pool_stats(visualnews_1m_cand_pool_path)

        # Reassign dids in the data
        for split in ["train", "val", "test"]:
            data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_{split}.jsonl")
            new_data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_new_{split}.jsonl")
            visualnews_data = load_jsonl_as_list(data_path)
            for visualnews_entry in visualnews_data:
                new_pos_cand_list = []
                for old_cand_did in visualnews_entry["pos_cand_list"]:
                    new_pos_cand_list.append(oldid_newdid_map[old_cand_did])
                visualnews_entry["pos_cand_list"] = new_pos_cand_list
            save_list_as_jsonl(visualnews_data, new_data_path, mode="w")

            # Print statistics
            total_entries, data = count_entries_in_file(new_data_path)
            print(f"MBEIR format Visualnews {split} data saved to {new_data_path}")
            print(f"Total number of entries in {new_data_path}: {total_entries}")
            visualnews_1m_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                visualnews_1m_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, visualnews_1m_cand_pool_dict)

    # Split the cand pool according to task
    if args.split_candidate_pool_by_task:
        print("Split the candidate pool according to task")
        # Load the 1M candidate pool
        visualnews_1m_cand_pool = load_jsonl_as_list(visualnews_1m_cand_pool_path)

        # Split the candidate pool according to task
        visualnews_task0_cand_pool = []
        visualnews_task3_cand_pool = []
        for visualnews_cand in visualnews_1m_cand_pool:
            if visualnews_cand["modality"] == "image":
                visualnews_task0_cand_pool.append(visualnews_cand)
            elif visualnews_cand["modality"] == "text":
                visualnews_task3_cand_pool.append(visualnews_cand)
            else:
                raise ValueError(f"Unknown modality: {visualnews_cand['modality']}")
        print(f"Task 0 candidate pool size: {len(visualnews_task0_cand_pool)}")
        print(f"Task 3 candidate pool size: {len(visualnews_task3_cand_pool)}")

        # Save the candidate pool
        visualnews_task0_cand_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_task0_cand_pool.jsonl")
        visualnews_task3_cand_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_task3_cand_pool.jsonl")
        save_list_as_jsonl(visualnews_task0_cand_pool, visualnews_task0_cand_pool_path)
        save_list_as_jsonl(visualnews_task3_cand_pool, visualnews_task3_cand_pool_path)
        print(f"Saved task 0 candidate pool to {visualnews_task0_cand_pool_path}")
        print(f"Saved task 3 candidate pool to {visualnews_task3_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(visualnews_task0_cand_pool_path)
        print_mbeir_format_cand_pool_stats(visualnews_task3_cand_pool_path)

    # Split the query data according to task
    if args.split_query_data_by_task:
        print("Split the query data according to task")
        visualnews_task0_cand_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_task0_cand_pool.jsonl")
        visualnews_task3_cand_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_task3_cand_pool.jsonl")

        for split in ["val", "test"]:
            data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_new_{split}.jsonl")
            task0_data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_task0_{split}.jsonl")
            task3_data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_task3_{split}.jsonl")

            # Load the data
            visualnews_data = load_jsonl_as_list(data_path)
            task0_data = []
            task3_data = []
            for entry in visualnews_data:
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
            visualnews_task0_cand_pool = load_mbeir_format_pool_file_as_dict(
                visualnews_task0_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(task0_data, visualnews_task0_cand_pool)
            print(f"Saved task 3 data to {task3_data_path}")
            visualnews_task3_cand_pool = load_mbeir_format_pool_file_as_dict(
                visualnews_task3_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(task3_data, visualnews_task3_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        visualnews_train_candidate_pool_path = os.path.join(visualnews_dir, "mbeir_visualnews_train_cand_pool.jsonl")
        mbeir_format_visualnews_train_data_path = os.path.join(visualnews_dir, f"mbeir_visualnews_new_train.jsonl")
        assert os.path.exists(
            mbeir_format_visualnews_train_data_path
        ), f"File {mbeir_format_visualnews_train_data_path} does not exist"

        # Load the training data
        visualnews_train_candidate_pool = {}
        visualnews_cand_pool = load_mbeir_format_pool_file_as_dict(
            visualnews_1m_cand_pool_path, doc_key_to_content=True, key_type="did"
        )
        print(f"Load candidate pool from {visualnews_1m_cand_pool_path}")
        mbeir_format_visualnews_train_data = load_jsonl_as_list(mbeir_format_visualnews_train_data_path)
        print(f"Load {len(mbeir_format_visualnews_train_data)} entries from {mbeir_format_visualnews_train_data_path}")

        for entry in mbeir_format_visualnews_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = visualnews_cand_pool[did]
                if did not in visualnews_train_candidate_pool:
                    visualnews_train_candidate_pool[did] = cand
                else:
                    if visualnews_train_candidate_pool[did] != cand:
                        print(
                            f"Duplicate did for two candidates found: {visualnews_train_candidate_pool[did]} and {cand}"
                        )

        # Save the training candidate pool
        visualnews_train_candidate_pool_list = list(visualnews_train_candidate_pool.values())
        visualnews_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(visualnews_train_candidate_pool_list, visualnews_train_candidate_pool_path)
        print(f"Saved training candidate pool to {visualnews_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(visualnews_train_candidate_pool_path)


if __name__ == "__main__":
    main()

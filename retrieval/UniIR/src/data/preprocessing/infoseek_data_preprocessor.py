"""
Infoseek_data_preprocessor.py
Module description:
    1. Convert all image files to the JPG format and resize the smaller dimension to 256.
    2. Generate candidate pool.
    3. Convert the dataset to the MBEIR format
"""

import argparse
import json
import os
import requests
import random

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
    load_mbeir_format_pool_file_as_dict,
    generate_mbeir_format_doc_key,
)

INFOSEEK_QUERY_MODALITY = "image,text"
INFOSEEK_CANDIDATE_MODALITY_IMAGE_TEXT = "image,text"
INFOSEEK_CANDIDATE_MODALITY_TEXT = "text"
INFOSEEK_DATASET_ID = get_dataset_id("INFOSEEK")
assert INFOSEEK_DATASET_ID is not None, "Unknown dataset name!"


def contains_answer(wikipedia_content, answer, answer_eval):
    # Checking answer strings
    for string in answer:
        if string in wikipedia_content:
            return True

    # Checking answer_eval strings
    for entry in answer_eval:
        if isinstance(entry, str) and entry in wikipedia_content:
            return True
    return False


def update_mbeir_format_infoseek_data_with_cand_pool(
    mbeir_format_infoseek_data_path,
    cand_pool_file_path,
):
    mbeir_format_infoseek_data = load_jsonl_as_list(mbeir_format_infoseek_data_path)

    def load_infoseek_mbeir_format_cand_pool_with_split_content_as_dict(
        cand_pool_file_path,
    ):
        """
        Load the raw candidate pool file into a dictionary.
        """
        can_dict = {}
        assert cand_pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."
        with open(cand_pool_file_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                src_content = json.loads(entry["src_content"])
                doc_key = src_content["wikidata_id"]
                if doc_key not in can_dict:
                    can_dict[doc_key] = [entry]
                else:
                    can_dict[doc_key].append(entry)
        return can_dict

    cand_pool_dict = load_infoseek_mbeir_format_cand_pool_with_split_content_as_dict(cand_pool_file_path)

    mbeir_entries = []
    for infoseek_data_entry in mbeir_format_infoseek_data:
        query_src_content = json.loads(infoseek_data_entry["query_src_content"])
        entity_id = query_src_content["entity_id"]
        answer = query_src_content["answer"]
        answer_eval = query_src_content["answer_eval"]
        potential_candidates = cand_pool_dict.get(entity_id, None)
        assert potential_candidates is not None, f"Missing candidates for entity_id {entity_id}"
        for potential_candidate in potential_candidates:
            wiki_string = potential_candidate["txt"]
            if contains_answer(wiki_string, answer, answer_eval):
                infoseek_data_entry["pos_cand_list"].append(potential_candidate["did"])
            else:
                infoseek_data_entry["neg_cand_list"].append(potential_candidate["did"])

        if len(infoseek_data_entry["pos_cand_list"]) > 0:
            mbeir_entries.append(infoseek_data_entry)
        # else:
        #     print(f"Warning: No positive candidate for infoseek entry {infoseek_data_entry}")

    # Generate query ID
    for i, entry in enumerate(mbeir_entries):
        entry.update({"qid": f"{INFOSEEK_DATASET_ID}:{i + 1}"})
    return mbeir_entries


def convert_raw_infoseek_cand_pool_to_mbeir_format_and_split_content(
    raw_cand_pool_file_path,
    mbeir_data_dir,
    include_src_content=True,
    skip_set=None,  # Used for augmenting the candidate pool
):
    assert raw_cand_pool_file_path.endswith(".jsonl"), f"Data Path {raw_cand_pool_file_path} is not a jsonl file"
    raw_infoseek_cand_list = load_jsonl_as_list(raw_cand_pool_file_path)
    output = []
    document_id = 1

    def get_directory_for_id(wikidata_id):
        if len(wikidata_id) > 4:  # Check if the ID is longer than 4 characters
            return wikidata_id[:4]  # Return the first four characters
        return wikidata_id  # Return the ID itself if it's shorter

    for raw_infoseek_cand in raw_infoseek_cand_list:
        wikidata_id = raw_infoseek_cand["wikidata_id"]
        if skip_set and wikidata_id in skip_set:
            continue  # Skip if the wikidata_id is in the skip set
        if raw_infoseek_cand.get("wikipedia_image_url", None):  # Check if the candidate has an image
            modality = INFOSEEK_CANDIDATE_MODALITY_IMAGE_TEXT
            oven_wiki_images_dir = os.path.join("mbeir_images", "oven_images", "wikipedia_images_full")
            img_dir = os.path.join(oven_wiki_images_dir, get_directory_for_id(wikidata_id))
            img_path = os.path.join(img_dir, f"{wikidata_id}.jpg")
            if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                print(f"Warning: Invalid image {img_path} for wikidata_id {wikidata_id}")
                modality = INFOSEEK_CANDIDATE_MODALITY_TEXT  # Set modality to text if the image is invalid
                img_path = None
        else:
            modality = INFOSEEK_CANDIDATE_MODALITY_TEXT
            img_path = None

        # Candidate text is the wikipedia content
        wiki_content = format_string(raw_infoseek_cand["wikipedia_content"])
        if not wiki_content:
            print(f"Warning: Empty wiki_content for wikidata_id {wikidata_id}")
            # print(f"raw_infoseek_cand: {raw_infoseek_cand}")
            continue  # Skip if the text is empty

        # Truncate the text to multiple string of 100 tokens
        def split_into_substrings(txt, token_limit=100):
            """Splits a string into substrings of up to token_limit tokens."""
            tokens = txt.split()
            substrings = []

            for i in range(0, len(tokens), token_limit):
                substring_tokens = tokens[i : i + token_limit]
                substrings.append(" ".join(substring_tokens))

            return substrings

        wiki_strings = split_into_substrings(wiki_content, 100)

        for wiki_string in wiki_strings:
            txt = f"{raw_infoseek_cand['wikipedia_title']}. {wiki_string}"
            txt = format_string(txt)
            candidate_pool_entry = {
                "txt": txt,
                "img_path": img_path,
                "modality": modality,
                "did": f"{INFOSEEK_DATASET_ID}:{document_id}",
            }
            if include_src_content:
                src_content = {
                    "wikidata_id": wikidata_id,
                    "wikipedia_title": raw_infoseek_cand["wikipedia_title"],
                }
                candidate_pool_entry["src_content"] = json.dumps(src_content)
            document_id += 1  # increment for next entry
            output.append(candidate_pool_entry)
    return output


def infoseek_to_mbeir_entry_and_create_raw_cand_pool(
    infoseek_entry,
    oven_cand_dict,
    infoseek_raw_cand_pool_file_path,
    kb_dict,
    qtype_dict,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert INFOSEEK data format to MBEIR format.
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
    mbeir_entry = {
        "qid": None,
        "query_txt": None,
        "query_img_path": None,
        "query_modality": INFOSEEK_QUERY_MODALITY,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    # Query text
    query_txt = format_string(infoseek_entry["question"])
    if not query_txt:
        print(f"Warning: Empty query text for infoseek entry {infoseek_entry}")
        return None  # Skip if the query text is empty
    mbeir_entry["query_txt"] = query_txt

    # Query image is stored in the same directory as the OVEN dataset
    img_subdir = infoseek_entry["image_id"][-8:-6]
    img_filename = f"{infoseek_entry['image_id']}.jpg"
    query_img_path = os.path.join("mbeir_images", "oven_images", img_subdir, img_filename)
    if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
        print(f"Warning: Invalid image {query_img_path} for oven entry {infoseek_entry}")
        return None  # Skip if the image is invalid
    mbeir_entry["query_img_path"] = query_img_path

    # Find the positive candidate
    entity_id = kb_dict.get(infoseek_entry["data_id"], None)
    pos_candidate = oven_cand_dict.get(entity_id, None)
    if not pos_candidate:
        print(f"Warning: No positive candidate for infoseek entry {infoseek_entry}")
        return None  # Skip if the positive candidate is missing

    # Query source content
    if include_src_content:
        query_src_content = {
            "data_id": infoseek_entry["data_id"],
            "answer": infoseek_entry["answer"],
            "answer_eval": infoseek_entry["answer_eval"],
            "data_split": infoseek_entry["data_split"],
            "entity_id": entity_id,
        }
        if qtype_dict:
            query_src_content["question_type"] = qtype_dict.get(infoseek_entry["data_id"], None)
            assert query_src_content["question_type"] is not None, "Question type is missing!"
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)

    # Positive candidate
    if contains_answer(
        pos_candidate["wikipedia_content"],
        infoseek_entry["answer"],
        infoseek_entry["answer_eval"],
    ):
        # Write the positive candidate to infoseek_cand_pool_file_path as raw candidates
        with open(infoseek_raw_cand_pool_file_path, "a") as f:
            f.write(json.dumps(pos_candidate) + "\n")
        return mbeir_entry
    else:
        # print(
        #     f"Warning: wikipedia_content does not contain answer through string matching for infoseek_entry {infoseek_entry['data_id']}")
        return None


def get_deduplicated_infoseek_data(infoseek_data):
    deduplicated_data = {}
    for infoseek_entry in infoseek_data:
        data_id = infoseek_entry["data_id"]
        if data_id not in deduplicated_data:
            deduplicated_data[data_id] = infoseek_entry
        else:
            print(f"\n Warning: Duplicate data entry: {data_id}")
            print(f"infoseek_entry: {infoseek_entry}")

    # Convert the dictionary values into a list
    return list(deduplicated_data.values())


def infoseek_to_mbeir_and_create_raw_cand_pool(
    infoseek_data,
    oven_wiki6m_file_path,
    infoseek_raw_cand_pool_file_path,
    kb_file_path,
    qtype_file_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    infoseek dataset to MBEIR format.
    """

    def load_infoseek_withkb_file_as_dict(withkb_file_path):
        """
        Load the Infoseek with kb file into a dictionary.
        Infoseek has unique data IDs, so we can use them as keys.
        """
        kb_dict = {}
        assert withkb_file_path.endswith(".jsonl"), "Only JSONL files are supported."
        with open(withkb_file_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                doc_key = entry["data_id"]
                kb_dict[doc_key] = entry["entity_id"]
        return kb_dict

    def load_qtype_file_as_dict(qtype_file_path):
        kb_dict = {}
        assert qtype_file_path.endswith(".jsonl"), "Only JSONL files are supported."
        with open(qtype_file_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                doc_key = entry["data_id"]
                kb_dict[doc_key] = entry["question_type"]
        return kb_dict

    def load_oven_wiki6m_file_as_dict(oven_wiki6m_file_path):
        """
        Load the Wiki6M_ver_1_0.jsonl file into a dictionary.
        """
        kb_dict = {}
        assert oven_wiki6m_file_path.endswith(".jsonl"), "Only JSONL files are supported."
        with open(oven_wiki6m_file_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                doc_key = entry["wikidata_id"]
                kb_dict[doc_key] = {
                    "wikidata_id": entry["wikidata_id"],
                    "wikipedia_title": entry["wikipedia_title"],
                    "wikipedia_content": entry["wikipedia_content"],
                    "wikipedia_image_url": entry.get("wikipedia_image_url", None),
                }
        return kb_dict

    # Load candidate pool
    oven_cand_dict = load_oven_wiki6m_file_as_dict(oven_wiki6m_file_path)
    print(f"loaded {len(oven_cand_dict)} candidates from {oven_wiki6m_file_path}")
    kb_dict = load_infoseek_withkb_file_as_dict(kb_file_path)
    print(f"loaded {len(kb_dict)} candidates from {kb_file_path}")
    qtype_dict = None
    if qtype_file_path:
        qtype_dict = load_qtype_file_as_dict(qtype_file_path)
    infoseek_data = get_deduplicated_infoseek_data(infoseek_data)
    print(f"loaded {len(infoseek_data)} entries from infoseek data")

    mbeir_entries = []
    for infoseek_entry in infoseek_data:
        mbeir_entry = infoseek_to_mbeir_entry_and_create_raw_cand_pool(
            infoseek_entry,
            oven_cand_dict,
            infoseek_raw_cand_pool_file_path,
            kb_dict,
            qtype_dict,
            mbeir_data_dir,
            include_src_content,
        )
        if mbeir_entry:  # Skip invalid entries
            mbeir_entries.append(mbeir_entry)
    return mbeir_entries


def compute_wiki6m_statistics(file_path):
    # Initialize counters and lists to hold lengths
    total_summary_length = 0
    total_content_length = 0
    num_entries = 0
    summary_lengths = []
    content_lengths = []

    with open(file_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())

            summary_length = len(entry["wikipedia_summary"].split()) if "wikipedia_summary" in entry else 0
            content_length = len(entry["wikipedia_content"].split()) if "wikipedia_content" in entry else 0

            total_summary_length += summary_length
            total_content_length += content_length

            summary_lengths.append(summary_length)
            content_lengths.append(content_length)

            num_entries += 1

    # Calculate statistics
    avg_summary_length = total_summary_length / num_entries
    avg_content_length = total_content_length / num_entries
    min_summary_length = min(summary_lengths)
    max_summary_length = max(summary_lengths)
    min_content_length = min(content_lengths)
    max_content_length = max(content_lengths)

    return {
        "avg_summary_length": avg_summary_length,
        "avg_content_length": avg_content_length,
        "min_summary_length": min_summary_length,
        "max_summary_length": max_summary_length,
        "min_content_length": min_content_length,
        "max_content_length": max_content_length,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description="Refactor Infoseek dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--infoseek_images_dir",
        type=str,
        default="mbeir_images/oven_images/",
        help="Relative directory path to save OVEN images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--oven_dir",
        type=str,
        default="src_data/oven",
        help="Relative directory path of OVEN files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--print_wiki6m_stats",
        action="store_true",
        help="Print statistics of the Wiki6M_ver_1_0.jsonl file.",
    )
    parser.add_argument(
        "--enable_cand_pool_and_to_mbeir_format_ph1",
        action="store_true",
        help="Enable converting infoseek data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_cand_pool_and_to_mbeir_format_ph2",
        action="store_true",
        help="Enable converting infoseek data to MBEIR format.",
    )
    parser.add_argument(
        "--augment_candidate_pool",
        action="store_true",
        help="Enable adding candidates to the candidate pool.",
    )
    parser.add_argument(
        "--remove_keys",
        action="store_true",
        help="Remove keys that are not supported by HuggingFace Datasets.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--assign_did_from_oven_cand_pool",
        action="store_true",
        help="Assign positive candidates from oven candidate pool to oven queries.",
    )
    parser.add_argument(
        "--split_val_into_val_and_test",
        action="store_true",
        help="Enable splitting the validation set into validation and test sets.",
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
    # Note: we keep the original project structure as in the OVEN dataset
    # So all the paths are hardcoded.
    oven_dir = os.path.join(args.mbeir_data_dir, args.oven_dir)
    infoseek_images_dir = os.path.join(args.mbeir_data_dir, args.infoseek_images_dir)
    infoseek_data_dir = os.path.join(oven_dir, "infoseek_data")
    infoseek_raw_cand_pool_file_path = os.path.join(oven_dir, "mbeir_infoseek_raw_cand_pool.jsonl")
    infoseek_cand_pool_file_path = os.path.join(oven_dir, "mbeir_infoseek_cand_pool.jsonl")

    oven_wiki6m_file_path = os.path.join(oven_dir, "Wiki6M_ver_1_0.jsonl")

    infoseek_train_file_path = os.path.join(infoseek_data_dir, "infoseek_train.jsonl")
    infoseek_train_withkb_file_path = os.path.join(infoseek_data_dir, "infoseek_train_withkb.jsonl")
    infoseek_val_file_path = os.path.join(infoseek_data_dir, "infoseek_val.jsonl")
    infoseek_val_withkb_file_path = os.path.join(infoseek_data_dir, "infoseek_val_withkb.jsonl")
    infoseek_val_qtype_file_path = os.path.join(infoseek_data_dir, "infoseek_val_qtype.jsonl")

    if args.print_wiki6m_stats:
        print("Printing statistics of the Wiki6M_ver_1_0.jsonl file...")
        stats = compute_wiki6m_statistics(oven_wiki6m_file_path)
        for key, value in stats.items():
            print(f"{key}: {value}")

    # Convert Infoseek data to MBEIR format Phase 1
    if args.enable_cand_pool_and_to_mbeir_format_ph1:
        print("Save infoseek candidate pool in infoseek format Phase 1")
        data_set_dict = {
            # TODO: Why we don't have a "qtype" file for training set?
            "train": [infoseek_train_file_path, infoseek_train_withkb_file_path],
            "val": [
                infoseek_val_file_path,
                infoseek_val_withkb_file_path,
                infoseek_val_qtype_file_path,
            ],
        }
        # Clean up infoseek_cand_pool_file_path if it exists
        with open(infoseek_raw_cand_pool_file_path, "w") as f:
            pass
        for split, data_paths in data_set_dict.items():
            mbeir_format_infoseek_data_path = os.path.join(oven_dir, f"mbeir_infoseek_{split}.jsonl")
            infoseek_file_path = data_paths[0]
            infoseek_withkb_file_path = data_paths[1]
            infoseek_qtype_file_path = data_paths[2] if len(data_paths) > 2 else None
            infoseek_data = load_jsonl_as_list(infoseek_file_path)
            mbeir_entries = infoseek_to_mbeir_and_create_raw_cand_pool(
                infoseek_data,
                oven_wiki6m_file_path,
                infoseek_raw_cand_pool_file_path,
                infoseek_withkb_file_path,
                infoseek_qtype_file_path,
                args.mbeir_data_dir,
                include_src_content=True,
            )
            save_list_as_jsonl(mbeir_entries, mbeir_format_infoseek_data_path, mode="w")

            # Print statistics
            total_entries, _data = count_entries_in_file(mbeir_format_infoseek_data_path)
            print(f"MBEIR format nights {split} data saved to {mbeir_format_infoseek_data_path}")
            print(f"Total number of entries in {mbeir_format_infoseek_data_path}: {total_entries}")
            # print_mbeir_format_dataset_stats(_data)

        # Clean duplicates in the raw candidate pool
        print("Clean duplicates in the candidate pool")
        infoseek_raw_cand_set = {}
        with open(infoseek_raw_cand_pool_file_path, "r") as f:
            for line in f:
                candidate = json.loads(line.strip())
                doc_key = candidate["wikidata_id"]
                if doc_key not in infoseek_raw_cand_set:
                    infoseek_raw_cand_set[doc_key] = candidate
        infoseek_raw_cand_pool = list(infoseek_raw_cand_set.values())

        # Save the raw candidate pool
        save_list_as_jsonl(infoseek_raw_cand_pool, infoseek_raw_cand_pool_file_path, mode="w")
        print(f"len(infoseek_cand_pool): {len(infoseek_raw_cand_pool)}")
        print(f"infoseek format candidate pool saved to {infoseek_raw_cand_pool_file_path}")

    # Convert Infoseek data to MBEIR format Phase 2
    if args.enable_cand_pool_and_to_mbeir_format_ph2:
        # Convert the raw candidate pool to MBEIR format and split the Wikipedia content into multiple strings
        cand_pool_entries = convert_raw_infoseek_cand_pool_to_mbeir_format_and_split_content(
            infoseek_raw_cand_pool_file_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        save_list_as_jsonl(cand_pool_entries, infoseek_cand_pool_file_path, mode="w")
        print(f"len(infoseek_cand_pool): {len(cand_pool_entries)}")
        print(f"MBEIR format candidate pool with content splitting saved to {infoseek_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(infoseek_cand_pool_file_path)

        # Trim training queries
        def load_mbeir_format_infoseek_pool_file_as_dict(pool_file_path, doc_key_to_content=False):
            pool_dict = {}
            assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."

            with open(pool_file_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    src_content = json.loads(entry["src_content"])
                    doc_key = src_content["wikidata_id"]
                    if doc_key_to_content:
                        pool_dict[doc_key] = entry
                    else:
                        pool_dict[doc_key] = entry["did"]
            return pool_dict

        def print_candidate_to_query_statistics(data, candidate_pool, description=""):
            # Count the number of `data_id` for each unique `entity_id`
            entity_counts = {}
            modality_counts = {"text": {}, "image,text": {}}

            for item in data:
                query_src_content = json.loads(item["query_src_content"])
                entity_id = query_src_content["entity_id"]
                modality = candidate_pool[entity_id]["modality"] if entity_id in candidate_pool else None

                if entity_id not in entity_counts:
                    entity_counts[entity_id] = []

                entity_counts[entity_id].append(query_src_content["data_id"])

                # Populate modality_counts
                if modality:
                    if entity_id not in modality_counts[modality]:
                        modality_counts[modality][entity_id] = []
                    modality_counts[modality][entity_id].append(query_src_content["data_id"])

            # Calculate and print general statistics
            counts = [len(data_ids) for _, data_ids in entity_counts.items()]
            avg_count = sum(counts) / len(counts)
            max_count = max(counts)
            min_count = min(counts)
            total_entity_ids = len(entity_counts)

            print(f"\nStatistics for {description}")
            print(f"Total number of data_id: {len(data)}")
            print(f"Average number of data_id per entity_id: {avg_count}")
            print(f"Maximum number of data_id for an entity_id: {max_count}")
            print(f"Minimum number of data_id for an entity_id: {min_count}")
            print(f"Total number of unique entity_id: {total_entity_ids}")

            # Calculate and print modality-specific statistics
            for modality, counts in modality_counts.items():
                if counts:
                    counts_list = [len(data_ids) for _, data_ids in counts.items()]
                    avg_count = sum(counts_list) / len(counts_list)
                    max_count = max(counts_list)
                    min_count = min(counts_list)

                    print(f"\nStatistics for modality: {modality}")
                    print(f"Total number of data_id: {sum(counts_list)}")
                    print(f"Average number of data_id per entity_id: {avg_count}")
                    print(f"Maximum number of data_id for an entity_id: {max_count}")
                    print(f"Minimum number of data_id for an entity_id: {min_count}")
                    print(f"Total number of unique entity_id: {len(counts)}")

        def get_entity_to_query_mapping(data_paths, candidate_pool):
            entity_to_query_mapping = {}
            for data_path in data_paths:
                with open(data_path, "r") as f:
                    data = [json.loads(line) for line in f]

                    # Print statistics before trimming
                    print_candidate_to_query_statistics(data, candidate_pool, description=data_path)
                    for entry in data:
                        query_src_content = json.loads(entry["query_src_content"])
                        entity_id = query_src_content["entity_id"]
                        modality = candidate_pool[entity_id]["modality"] if entity_id in candidate_pool else None
                        if entity_id not in entity_to_query_mapping:
                            entity_to_query_mapping[entity_id] = {
                                "queries": [],
                                "modality": modality,
                            }
                        entity_to_query_mapping[entity_id]["queries"].append(entry)

            all_queries = [query for entry in entity_to_query_mapping.values() for query in entry["queries"]]
            print_candidate_to_query_statistics(all_queries, candidate_pool, description="before trimming")
            return entity_to_query_mapping

        def trim_queries(entity_to_query_mapping, text_threshold, image_text_threshold):
            trimmed_entity_to_query_mapping = {}
            for entity_id, entry in entity_to_query_mapping.items():
                query_list = entry["queries"]
                modality = entry["modality"]

                # Determine the threshold based on modality
                threshold = image_text_threshold if modality == "image,text" else text_threshold

                random.shuffle(query_list)
                if len(query_list) <= threshold:
                    trimmed_entity_to_query_mapping[entity_id] = query_list
                else:
                    trimmed_entity_to_query_mapping[entity_id] = query_list[:threshold]
            return trimmed_entity_to_query_mapping

        def save_trimmed_data(
            trimmed_entity_to_query_mapping,
            original_data_files,
            trimmed_data_files,
            candidate_pool,
        ):
            for data_path, trimmed_data_path in zip(original_data_files, trimmed_data_files):
                trimmed_data = []
                with open(data_path, "r") as f:
                    original_data = [json.loads(line) for line in f]
                    for entry in original_data:
                        query_src_content = json.loads(entry["query_src_content"])
                        entity_id = query_src_content["entity_id"]
                        if (
                            entity_id in trimmed_entity_to_query_mapping
                            and entry in trimmed_entity_to_query_mapping[entity_id]
                        ):
                            trimmed_data.append(entry)
                            trimmed_entity_to_query_mapping[entity_id].remove(entry)

                save_list_as_jsonl(trimmed_data, trimmed_data_path, mode="w")
                print(f"\nTrimmed data saved to {trimmed_data_path}")
                print_candidate_to_query_statistics(trimmed_data, candidate_pool, description=trimmed_data_path)

        candidate_pool_dict = load_mbeir_format_infoseek_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True
        )
        mbier_format_infoseek_train_file_path = os.path.join(oven_dir, "mbeir_infoseek_train.jsonl")
        trimmed_mbeir_format_infoseek_data_path = os.path.join(oven_dir, "mbeir_infoseek_train_trimmed.jsonl")
        entity_to_query_mapping = get_entity_to_query_mapping(
            [mbier_format_infoseek_train_file_path], candidate_pool_dict
        )
        trimmed_entity_to_query_mapping = trim_queries(entity_to_query_mapping, 80, 200)
        save_trimmed_data(
            trimmed_entity_to_query_mapping,
            [mbier_format_infoseek_train_file_path],
            [trimmed_mbeir_format_infoseek_data_path],
            candidate_pool_dict,
        )

        data_splits_paths = [
            ("train", trimmed_mbeir_format_infoseek_data_path),
            ("val", os.path.join(oven_dir, f"mbeir_infoseek_val.jsonl")),
        ]
        for split, data_path in data_splits_paths:
            mbeir_entries = update_mbeir_format_infoseek_data_with_cand_pool(
                data_path,
                infoseek_cand_pool_file_path,
            )
            data_path_final = os.path.join(oven_dir, f"mbeir_infoseek_{split}_final.jsonl")
            save_list_as_jsonl(mbeir_entries, data_path_final, mode="w")

            # Print statistics
            total_entries, _data = count_entries_in_file(data_path_final)
            print(f"MBEIR format INFOSEEK {split} data saved to {data_path_final}")
            print(f"Total number of entries in {data_path_final}: {total_entries}")
            infoseek_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(_data, infoseek_cand_pool_dict)

    # Add more candidates to the candidate pool
    if args.augment_candidate_pool:
        print("Adding more candidates to the candidate pool")

        def load_mbeir_format_infoseek_pool_file_as_set(pool_file_path):
            pool_set = set()
            assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."
            with open(pool_file_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    src_content = json.loads(entry["src_content"])
                    doc_key = src_content["wikidata_id"]
                    pool_set.add(doc_key)
            return pool_set

        skip_pool_set = load_mbeir_format_infoseek_pool_file_as_set(infoseek_cand_pool_file_path)

        # document_id_start set to the last document id in the candidate pool
        def count_lines_in_file(file_path):
            with open(file_path, "r") as f:
                return sum(1 for _ in f)

        document_id_start = count_lines_in_file(infoseek_cand_pool_file_path) + 1
        print(f"document_id_start: {document_id_start}")

        oven_wiki6m_mbeir_format_pool_with_split_content = (
            convert_raw_infoseek_cand_pool_to_mbeir_format_and_split_content(
                oven_wiki6m_file_path,
                args.mbeir_data_dir,
                include_src_content=True,
                skip_set=skip_pool_set,
            )
        )

        # Random sample 1M candidates
        random.shuffle(oven_wiki6m_mbeir_format_pool_with_split_content)
        augment_size = 1000000
        oven_wiki6m_mbeir_format_pool_with_split_content = oven_wiki6m_mbeir_format_pool_with_split_content[
            :augment_size
        ]

        # Reassign document ids
        for i, entry in enumerate(oven_wiki6m_mbeir_format_pool_with_split_content):
            entry["did"] = f"{INFOSEEK_DATASET_ID}:{document_id_start + i}"

        # Append the new candidates to the candidate pool
        save_list_as_jsonl(
            oven_wiki6m_mbeir_format_pool_with_split_content,
            infoseek_cand_pool_file_path,
            mode="a",
        )
        print(f"{augment_size} candidates added to the candidate pool")
        print_mbeir_format_cand_pool_stats(infoseek_cand_pool_file_path)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        infoseek_train_candidate_pool_path = os.path.join(oven_dir, "mbeir_infoseek_train_cand_pool.jsonl")
        mbeir_format_infoseek_train_data_path = os.path.join(oven_dir, f"mbeir_infoseek_train_final.jsonl")
        assert os.path.exists(
            mbeir_format_infoseek_train_data_path
        ), f"File {mbeir_format_infoseek_train_data_path} does not exist"

        # Load the training data
        infoseek_train_candidate_pool = {}
        infoseek_cand_pool = load_mbeir_format_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_infoseek_train_data = load_jsonl_as_list(mbeir_format_infoseek_train_data_path)
        for entry in mbeir_format_infoseek_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = infoseek_cand_pool[did]
                if did not in infoseek_train_candidate_pool:
                    infoseek_train_candidate_pool[did] = cand
                else:
                    if infoseek_train_candidate_pool[did] != cand:
                        print(
                            f"Duplicate did for two candidates found: {infoseek_train_candidate_pool[did]} and {cand}"
                        )

        # Save the training candidate pool
        infoseek_train_candidate_pool_list = list(infoseek_train_candidate_pool.values())
        infoseek_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(infoseek_train_candidate_pool_list, infoseek_train_candidate_pool_path)
        print(f"Saved training candidate pool to {infoseek_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(infoseek_train_candidate_pool_path)

    # Save the training candidate pool in MBEIR format
    if args.assign_did_from_oven_cand_pool:
        # Assign infoseek did to the oven data
        print("Assigning positive candidates from oven candidate pool to infoseek queries...")
        oven_1m_candidate_pool_path = os.path.join(oven_dir, "mbeir_oven_1m_cand_pool.jsonl")
        oven_train_candidate_pool_path = os.path.join(oven_dir, "mbeir_oven_train_cand_pool.jsonl")

        infoseek_cand_dict = load_mbeir_format_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        for split, oven_cand_pool_path in [
            ("train", oven_train_candidate_pool_path),
            ("val", oven_1m_candidate_pool_path),
        ]:
            oven_cand_pool = load_mbeir_format_pool_file_as_dict(oven_cand_pool_path, key_type="mbeir_converted_key")
            mbeir_format_infoseek_data_path = os.path.join(oven_dir, f"mbeir_infoseek_{split}_final.jsonl")
            infoseek_data = load_jsonl_as_list(mbeir_format_infoseek_data_path)
            for infoseek_entry in infoseek_data:
                oven_dids = []
                for did in infoseek_entry["pos_cand_list"]:
                    infoseek_cand = infoseek_cand_dict[did]
                    doc_key = generate_mbeir_format_doc_key(infoseek_cand)
                    oven_cand_did = oven_cand_pool.get(doc_key, None)
                    if oven_cand_did:
                        oven_dids.append(oven_cand_did)
                infoseek_entry["pos_cand_list"].extend(oven_dids)

            mbeir_format_infoseek_data_merged_path = os.path.join(oven_dir, f"mbeir_infoseek_{split}_merged.jsonl")
            save_list_as_jsonl(infoseek_data, mbeir_format_infoseek_data_merged_path, mode="w")

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_infoseek_data_merged_path)
            print(f"MBEIR format Infoseek {split} data saved to {mbeir_format_infoseek_data_merged_path}")
            print(f"Total number of entries in {mbeir_format_infoseek_data_merged_path}: {total_entries}")

            # Build combined pool
            oven_cand_pool = load_mbeir_format_pool_file_as_dict(
                oven_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            combined_pool_dict = {**oven_cand_pool, **infoseek_cand_dict}
            print_mbeir_format_dataset_stats(data, combined_pool_dict)

    # Split val set into val and test
    if args.split_val_into_val_and_test:
        mbeir_infoseek_val_data_path = os.path.join(oven_dir, "mbeir_infoseek_val_merged.jsonl")
        print(f"Splitting {mbeir_infoseek_val_data_path} into validation and test sets...")
        mbeir_infoseek_val_data = load_jsonl_as_list(mbeir_infoseek_val_data_path)
        random.seed(2023)
        random.shuffle(mbeir_infoseek_val_data)
        new_infoseek_val_data = mbeir_infoseek_val_data[: len(mbeir_infoseek_val_data) // 2]
        new_infoseek_test_data = mbeir_infoseek_val_data[len(mbeir_infoseek_val_data) // 2 :]
        mbeir_infoseek_new_val_data_path = os.path.join(oven_dir, "mbeir_infoseek_new_val.jsonl")
        mbeir_infoseek_new_test_data_path = os.path.join(oven_dir, "mbeir_infoseek_new_test.jsonl")
        oven_1m_candidate_pool_path = os.path.join(oven_dir, "mbeir_oven_1m_cand_pool.jsonl")
        oven_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            oven_1m_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        infoseek_cand_pool_file_path = os.path.join(oven_dir, "mbeir_infoseek_cand_pool.jsonl")
        infoseek_cand_pool = load_mbeir_format_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        combined_pool_dict = {**infoseek_cand_pool, **oven_cand_pool_dict}
        print(f"Saved new validation data to {mbeir_infoseek_new_val_data_path}")
        save_list_as_jsonl(new_infoseek_val_data, mbeir_infoseek_new_val_data_path, mode="w")
        print_mbeir_format_dataset_stats(new_infoseek_val_data, combined_pool_dict)
        print(f"Saved new test data to {mbeir_infoseek_new_test_data_path}")
        save_list_as_jsonl(new_infoseek_test_data, mbeir_infoseek_new_test_data_path, mode="w")
        print_mbeir_format_dataset_stats(new_infoseek_test_data, combined_pool_dict)

    # Split candidate pool by task
    # Split the cand pool according to task
    if args.split_candidate_pool_by_task:
        # Load the candidate pool
        infoseek_cand_pool = load_jsonl_as_list(infoseek_cand_pool_file_path)
        print(f"Split the candidate pool {infoseek_cand_pool_file_path} according to task")

        # Split the candidate pool
        infoseek_task6_cand_pool = []
        infoseek_task8_cand_pool = []
        for infoseek_cand in infoseek_cand_pool:
            if infoseek_cand["modality"] == "text":
                infoseek_task6_cand_pool.append(infoseek_cand)
            elif infoseek_cand["modality"] == "image,text":
                infoseek_task8_cand_pool.append(infoseek_cand)
            else:
                raise ValueError(f"Unknown modality: {infoseek_cand['modality']}")
        print(f"Number of candidates for task 6: {len(infoseek_task6_cand_pool)}")
        print(f"Number of candidates for task 8: {len(infoseek_task8_cand_pool)}")

        # Save the candidate pool
        infoseek_task6_cand_pool_path = os.path.join(oven_dir, "mbeir_infoseek_task6_cand_pool.jsonl")
        infoseek_task8_cand_pool_path = os.path.join(oven_dir, "mbeir_infoseek_task8_cand_pool.jsonl")
        save_list_as_jsonl(infoseek_task6_cand_pool, infoseek_task6_cand_pool_path)
        save_list_as_jsonl(infoseek_task8_cand_pool, infoseek_task8_cand_pool_path)
        print(f"Saved task 6 candidate pool to {infoseek_task6_cand_pool_path}")
        print(f"Saved task 8 candidate pool to {infoseek_task8_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(infoseek_task6_cand_pool_path)
        print_mbeir_format_cand_pool_stats(infoseek_task8_cand_pool_path)

    # Split the query data according to task
    if args.split_query_data_by_task:
        print("Split the query data according to task")
        infoseek_cand_dict = load_mbeir_format_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        oven_1m_candidate_pool_path = os.path.join(oven_dir, "mbeir_oven_1m_cand_pool.jsonl")
        oven_1m_cand_dict = load_mbeir_format_pool_file_as_dict(
            oven_1m_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )

        for split in ["val", "test"]:
            infoseek_data_path = os.path.join(oven_dir, f"mbeir_infoseek_new_{split}.jsonl")
            task6_data_path = os.path.join(oven_dir, f"mbeir_infoseek_task6_{split}.jsonl")
            task8_data_path = os.path.join(oven_dir, f"mbeir_infoseek_task8_{split}.jsonl")

            # Load the data
            infoseek_data = load_jsonl_as_list(infoseek_data_path)
            task6_data = []
            task8_data = []
            for entry in infoseek_data:
                pos_cand_did = entry["pos_cand_list"][0]
                pos_cand_modality = infoseek_cand_dict[pos_cand_did]["modality"]
                if pos_cand_modality == "text":
                    task6_data.append(entry)
                elif pos_cand_modality == "image,text":
                    task8_data.append(entry)
                else:
                    raise ValueError(f"Unknown modality: {entry['query_modality']}")

            # Save the data
            save_list_as_jsonl(task6_data, task6_data_path)
            save_list_as_jsonl(task8_data, task8_data_path)

            combined_pool_dict = {**infoseek_cand_dict, **oven_1m_cand_dict}
            print(f"Saved task 6 data to {task6_data_path}")
            print_mbeir_format_dataset_stats(task6_data, combined_pool_dict)
            print(f"Saved task 8 data to {task8_data_path}")
            print_mbeir_format_dataset_stats(task8_data, combined_pool_dict)


if __name__ == "__main__":
    main()

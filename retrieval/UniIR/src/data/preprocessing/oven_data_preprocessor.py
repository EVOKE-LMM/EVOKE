"""
Oven_data_preprocessor.py

Module description:
    1. Download images from Wikipedia to enrich the oven dataset.
        The script fetches images based on URLs fields from the dataset.
    2. Convert all image files to the JPG format and resize the smaller dimension to 256.
    3. Generate candidate pool.
    4. Convert the dataset to the MBEIR format.
"""

import os
import json
import requests
from PIL import Image
from io import BytesIO
import random

# import cairosvg
import argparse
import xml.etree.ElementTree as ET

from multiprocessing import Pool, cpu_count, Manager, Lock

from utils import (
    parallel_process_image_directory,
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

OVEN_DATASET_ID = get_dataset_id("OVEN")
assert OVEN_DATASET_ID is not None, "Unknown dataset name!"


def oven_to_mbeir_entry(
    oven_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert OVEN data format to MBEIR format.
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
        "query_txt": format_string(oven_entry["question"]),
        "query_img_path": None,
        "query_modality": "image,text",
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    # Locate the image path according to the original OVEN data structure
    img_subdir = oven_entry["image_id"][-8:-6]
    img_filename = f"{oven_entry['image_id']}.jpg"
    query_img_path = os.path.join("mbeir_images", "oven_images", img_subdir, img_filename)
    if not is_valid_image(os.path.join(mbeir_data_dir, query_img_path)):
        print(f"Warning: Invalid image {query_img_path} for oven entry {oven_entry}")
        return None  # Skip if the image is invalid

    mbeir_entry["query_img_path"] = query_img_path

    # Add positive candidate
    pos_candidate_did = candidate_pool.get(oven_entry["entity_id"], None)
    if not pos_candidate_did:
        print(f"Warning: No positive candidate for oven entry {oven_entry}")
        return None  # Skip if the positive candidate is missing
    mbeir_entry["pos_cand_list"].append(pos_candidate_did)  # Add the positive candidate

    # Add source content
    if include_src_content:
        query_src_content = {
            "data_id": oven_entry["data_id"],
            "image_id": oven_entry["image_id"],
            "entity_id": oven_entry["entity_id"],  # We don't include "entity_text"
            "data_split": oven_entry["data_split"],
        }
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)
    return mbeir_entry


def load_mbeir_format_oven_pool_file_as_dict(pool_file_path, doc_key_to_content=False):
    """
    Load the OVEN candidate pool file into a dictionary.
    OVEN has unique candidate IDs, so we can use them as keys.
    """
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


def process_oven_data_chunk(args):
    chunk, candidate_pool, mbeir_data_dir, include_src_content = args
    mbeir_chunk_entries = []
    for oven_entry in chunk:
        mbeir_entry = oven_to_mbeir_entry(
            oven_entry,
            candidate_pool,
            mbeir_data_dir,
            include_src_content=include_src_content,
        )
        if mbeir_entry:  # Skip invalid entries
            mbeir_chunk_entries.append(mbeir_entry)
    return mbeir_chunk_entries


def get_deduplicated_oven_data(oven_data):
    deduplicated_data = []
    seen_data = set()
    for oven_entry in oven_data:
        data_id = oven_entry["data_id"]
        if data_id not in seen_data:
            deduplicated_data.append(oven_entry)
            seen_data.add(data_id)
    return deduplicated_data


def parallel_oven_to_mbeir(oven_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    # Load candidate pool
    with Manager() as manager:
        candidate_pool = manager.dict(load_mbeir_format_oven_pool_file_as_dict(candidate_pool_file_path))

        # Deduplicated data
        num_processes = cpu_count() // 2
        deduplicated_data = get_deduplicated_oven_data(oven_data)
        data_chunks = split_into_chunks(deduplicated_data, num_processes)

        args = [(data_chunks[i], candidate_pool, mbeir_data_dir, include_src_content) for i in range(num_processes)]

        # Use multiprocessing to process chunks in parallel
        with Pool(num_processes) as p:
            results = p.map(process_oven_data_chunk, args)

        # Merge result
        mbeir_entries = [entry for chunk_result in results for entry in chunk_result]

        # Generate query ID
        for i, entry in enumerate(mbeir_entries):
            entry.update({"qid": f"{OVEN_DATASET_ID}:{i + 1}"})
        return mbeir_entries


def split_into_chunks(data, num_chunks):
    avg = len(data) // num_chunks
    remainder = len(data) % num_chunks
    chunks = [data[i * avg : i * avg + avg] for i in range(num_chunks)]
    if remainder:
        chunks[-1] += data[-remainder:]  # Append the remainder to the last chunk
    return chunks


def get_deduplicated_data(data):
    deduplicated_data = []
    seen_candidates = set()
    for oven_cand in data:
        wikidata_id = oven_cand["wikidata_id"]
        if wikidata_id not in seen_candidates:
            deduplicated_data.append(oven_cand)
            seen_candidates.add(wikidata_id)
    return deduplicated_data


def get_directory_for_id(wikidata_id):
    if len(wikidata_id) > 4:  # Check if the ID is longer than 4 characters
        return wikidata_id[:4]  # Return the first four characters
    return wikidata_id  # Return the ID itself if it's shorter


def truncate_summary_to_max_tokens(summary, max_tokens=100):
    tokens = summary.split()  # Splitting by spaces to get words
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]  # Keeping only the first 100 words
    return " ".join(tokens)  # Joining tokens back to string


def generate_oven_candidate_pool_chunk(args):
    oven_cand_data, start_id, mbeir_data_dir, include_src_content = args
    document_id = start_id
    output = []
    for oven_cand in oven_cand_data:
        wikidata_id = oven_cand["wikidata_id"]
        if oven_cand.get("wikipedia_image_url", None):  # Check if the candidate has an image
            modality = "image,text"
            oven_wiki_images_dir = os.path.join("mbeir_images", "oven_images", "wikipedia_images_full")
            img_dir = os.path.join(oven_wiki_images_dir, get_directory_for_id(wikidata_id))
            img_path = os.path.join(img_dir, f"{wikidata_id}.jpg")
            if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                print(f"Warning: Invalid image {img_path} for wikidata_id {wikidata_id}")
                modality = "text"  # Set modality to text if the image is invalid
                img_path = None
        else:
            modality = "text"
            img_path = None

        # Candidate text is a concatenation of the title and content
        # Note many candidates don't have a summary and content
        wikipedia_content = oven_cand.get("wikipedia_content", "")
        truncated_content = truncate_summary_to_max_tokens(wikipedia_content)
        txt = f"{oven_cand['wikipedia_title']}. {truncated_content}"
        txt = format_string(txt)
        if not txt:
            print(f"Warning: Empty text for wikidata_id {wikidata_id}")
            continue  # Skip if the text is empty

        # Add the candidate to the pool
        candidate_pool_entry = {
            "txt": txt,
            "img_path": img_path,
            "modality": modality,
            "did": f"{OVEN_DATASET_ID}:{document_id}",
        }
        if include_src_content:
            src_content = {
                "wikidata_id": wikidata_id,
                "wikipedia_title": oven_cand.get("wikipedia_title", ""),
            }
            candidate_pool_entry["src_content"] = json.dumps(src_content)
        document_id += 1  # increment for next entry
        output.append(candidate_pool_entry)
    return output


def parallel_generate_oven_candidate_pool(
    oven_wiki6m_file_path,
    oven_candidate_pool_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate OVEN candidate pool in mbeir format and save it to a jsonl file.
    Here is the format of expected Wiki6M_ver_1_0.jsonl file:
    {
    "wikidata_id": "Q3783438",
    "wikipedia_title": "Harold Keeling",
    "wikipedia_content": "Harold A. Keeling...",
    "wikipedia_image_url": null,
    "wikipedia_summary": "Harold A. Keeling...",
    }
    """
    # oven_wiki6m_file_path is a jsonl file
    assert oven_wiki6m_file_path.endswith(".jsonl"), f"Data Path {oven_wiki6m_file_path} is not a jsonl file"
    oven_cand_data = load_jsonl_as_list(oven_wiki6m_file_path)

    # Split data into chunks
    num_processes = cpu_count() // 4
    deduplicated_data = get_deduplicated_data(oven_cand_data)
    data_chunks = split_into_chunks(deduplicated_data, num_processes)

    # Compute starting IDs for all chunks
    start_ids = [1]
    for i in range(1, num_processes):
        start_ids.append(start_ids[-1] + len(data_chunks[i - 1]))
    args = [(data_chunks[i], start_ids[i], mbeir_data_dir, include_src_content) for i in range(num_processes)]

    # Use multiprocessing to process chunks in parallel
    with Pool(num_processes) as p:
        results = p.map(generate_oven_candidate_pool_chunk, args)

    # Merge results and write to file
    with open(oven_candidate_pool_path, "w") as outfile:
        for chunk_result in results:
            for entry in chunk_result:
                outfile.write(json.dumps(entry) + "\n")


def sanitize_svg(svg_content):
    # Parse the SVG content
    tree = ET.ElementTree(ET.fromstring(svg_content))
    root = tree.getroot()

    # Remove entities (like <!ENTITY ...>)
    for elem in list(root):
        if not isinstance(elem.tag, str):  # this checks for entities
            root.remove(elem)

    # Convert the modified SVG back to string
    return ET.tostring(root, encoding="utf-8").decode("utf-8")


def download_wiki_images(mbeir_data_dir, oven_dir):
    oven_wiki_img_dir = os.path.join(mbeir_data_dir, "mbeir_images/oven_images/wiki_images/")
    oven_data_dir = os.path.join(oven_dir, "oven_data/")
    # Load the Wikipedia data to map entity IDs to image URLs
    with open(os.path.join(oven_dir, "Wiki6M_ver_1_0.jsonl"), "r") as f:
        wikidata_to_url = {
            entry["wikidata_id"]: entry.get("wikipedia_image_url") for entry in (json.loads(line) for line in f)
        }

    # List of files to process
    files_to_process = [
        "oven_entity_train.jsonl",
        "oven_entity_val.jsonl",
        "oven_query_train.jsonl",
        "oven_query_val.jsonl",
    ]
    files_to_process = [os.path.join(oven_data_dir, file_name) for file_name in files_to_process]

    # Check and create target directory for images if it doesn't exist
    if not os.path.exists(oven_wiki_img_dir):
        os.makedirs(oven_wiki_img_dir)

    entity_id_to_img_path = {}

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    for file_name in files_to_process:
        # Read the file
        with open(file_name, "r") as f:
            entries = [json.loads(line) for line in f]

        updated_entries = []

        for entry in entries:
            entity_id = entry["entity_id"]
            img_url = wikidata_to_url.get(entity_id, None)

            relative_img_path = os.path.join(oven_wiki_img_dir, f"{entity_id}.jpg")  # This remains relative

            if img_url:
                full_img_path = os.path.join(oven_wiki_img_dir, f"{entity_id}.jpg")

                if entity_id in entity_id_to_img_path:
                    entry["wiki_img_path"] = entity_id_to_img_path[entity_id]
                else:
                    response = requests.get(img_url, headers=headers, stream=True)

                    if response.status_code == 200:
                        try:
                            if img_url.lower().endswith(".svg"):
                                sanitized_svg_content = sanitize_svg(response.content.decode("utf-8"))
                                png_output = BytesIO()
                                cairosvg.svg2png(
                                    bytestring=sanitized_svg_content.encode("utf-8"),
                                    write_to=png_output,
                                )
                                image = Image.open(png_output)
                            else:
                                image = Image.open(response.raw)

                            image.convert("RGB").save(full_img_path, "JPEG")
                            entry["wiki_img_path"] = relative_img_path
                            entity_id_to_img_path[entity_id] = relative_img_path
                        except Exception as e:
                            print(f"Error processing image from URL: {img_url}. Error: {e}")
                            entry["wiki_img_path"] = None
                    else:
                        print(f"Failed to download image from URL: {img_url}")
                        entry["wiki_img_path"] = None
            else:
                entry["wiki_img_path"] = None

            updated_entries.append(entry)

        # Write the updated entries back to the file
        with open(file_name, "w") as f:
            for entry in updated_entries:
                f.write(json.dumps(entry) + "\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format OVEN images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--oven_images_dir",
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
        "--download_wiki_images",
        action="store_true",
        help=" Download images from Wikipedia to enrich the oven dataset.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating OVEN candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_trim_training_queries",
        action="store_true",
        help="Enable trimming the number of queries in the training sets.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting OVEN data to MBEIR format.",
    )
    parser.add_argument(
        "--trim_candidate_pool",
        action="store_true",
        help="Enable trimming the number of candidates in the candidate pool.",
    )
    parser.add_argument(
        "--enable_training_candidate_pool",
        action="store_true",
        help="Enable generating training candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--assign_did_from_infoseek_cand_pool",
        action="store_true",
        help="Assign positive candidates from infoseek candidate pool to oven queries.",
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
    oven_images_dir = os.path.join(args.mbeir_data_dir, args.oven_images_dir)
    oven_candidate_pool_1m_path = os.path.join(oven_dir, "mbeir_oven_1m_cand_pool.jsonl")
    oven_candidate_pool_6m_path = os.path.join(oven_dir, "mbeir_oven_6m_cand_pool.jsonl")
    oven_wiki6m_file_path = os.path.join(oven_dir, "Wiki6M_ver_1_0.jsonl")
    oven_data_dir = os.path.join(oven_dir, "oven_data")
    oven_entity_train_file_path = os.path.join(oven_data_dir, "oven_entity_train.jsonl")
    oven_query_train_file_path = os.path.join(oven_data_dir, "oven_query_train.jsonl")
    oven_entity_train_trimmed_file_path = os.path.join(oven_data_dir, "oven_entity_train_trimmed.jsonl")
    oven_query_train_trimmed_file_path = os.path.join(oven_data_dir, "oven_query_train_trimmed.jsonl")
    oven_entity_val_file_path = os.path.join(oven_data_dir, "oven_entity_val.jsonl")
    oven_query_val_file_path = os.path.join(oven_data_dir, "oven_query_val.jsonl")

    if args.download_wiki_images:
        print("We have wikipedia_images_full now. No need to download.")
        download_wiki_images(args.mbeir_data_dir, oven_dir)

    if args.enable_image_processing:
        print(f"Processing images in {oven_images_dir}")
        parallel_process_image_directory(oven_images_dir)

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating OVEN 6M candidate pool in mbeir format...")
        parallel_generate_oven_candidate_pool(
            oven_wiki6m_file_path,
            oven_candidate_pool_6m_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"OVEN 6M Candidate pool saved to {oven_candidate_pool_6m_path}")
        print_mbeir_format_cand_pool_stats(oven_candidate_pool_6m_path)  # Print statistics

    # Trim the number of queries in the training sets
    if args.enable_trim_training_queries:

        def calculate_and_print_statistics(data, file_path, candidate_pool):
            # Count the number of `data_id` for each unique `entity_id`
            entity_counts = {}
            modality_counts = {"text": {}, "image,text": {}}

            for item in data:
                entity_id = item["entity_id"]
                modality = candidate_pool[entity_id]["modality"] if entity_id in candidate_pool else None

                if entity_id not in entity_counts:
                    entity_counts[entity_id] = []

                entity_counts[entity_id].append(item["data_id"])

                # Populate modality_counts
                if modality:
                    if entity_id not in modality_counts[modality]:
                        modality_counts[modality][entity_id] = []
                    modality_counts[modality][entity_id].append(item["data_id"])

            # Calculate and print general statistics
            counts = [len(data_ids) for _, data_ids in entity_counts.items()]
            avg_count = sum(counts) / len(counts)
            max_count = max(counts)
            min_count = min(counts)
            total_entity_ids = len(entity_counts)

            print(f"\nFor file {file_path}:")
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

        def consolidate_data(file_paths, candidate_pool_dict):
            consolidated_data = {}
            for path in file_paths:
                with open(path, "r") as f:
                    data = [json.loads(line) for line in f]

                    # Print statistics before trimming
                    calculate_and_print_statistics(data, path, candidate_pool_dict)

                    for item in data:
                        entity_id = item["entity_id"]
                        modality = (
                            candidate_pool_dict[entity_id]["modality"] if entity_id in candidate_pool_dict else None
                        )
                        if entity_id not in consolidated_data:
                            consolidated_data[entity_id] = {
                                "data": [],
                                "modality": modality,
                            }
                        consolidated_data[entity_id]["data"].append(item)

            all_items = [item for entry in consolidated_data.values() for item in entry["data"]]
            calculate_and_print_statistics(all_items, "consolidated data before trimming", candidate_pool_dict)
            return consolidated_data

        def trim_consolidated_data(consolidated, text_threshold, image_text_threshold):
            trimmed = {}
            for entity, entry in consolidated.items():
                data_list = entry["data"]
                modality = entry["modality"]

                # Determine the threshold based on modality
                threshold = image_text_threshold if modality == "image,text" else text_threshold

                random.shuffle(data_list)

                if len(data_list) <= threshold:
                    trimmed[entity] = data_list
                else:
                    trimmed[entity] = data_list[:threshold]

            total_data_ids = sum(len(data_list) for data_list in trimmed.values())
            print(f"Total number of data_id post-trimming: {total_data_ids}")
            return trimmed

        def save_trimmed_data(trimmed_data, original_files, trimmed_files, candidate_pool_dict):
            for file_path, trimmed_file_path in zip(original_files, trimmed_files):
                output_data = []
                with open(file_path, "r") as f:
                    original_data = [json.loads(line) for line in f]
                    for item in original_data:
                        entity_id = item["entity_id"]
                        if entity_id in trimmed_data and item in trimmed_data[entity_id]:
                            output_data.append(item)
                            trimmed_data[entity_id].remove(item)

                with open(trimmed_file_path, "w") as f:
                    for item in output_data:
                        f.write(json.dumps(item) + "\n")
                print(f"Trimmed data saved to {trimmed_file_path}")

                # Print statistics after trimming
                calculate_and_print_statistics(output_data, trimmed_file_path, candidate_pool_dict)

        candidate_pool_dict = load_mbeir_format_oven_pool_file_as_dict(
            oven_candidate_pool_6m_path, doc_key_to_content=True
        )
        file_paths = [oven_entity_train_file_path, oven_query_train_file_path]
        trimmed_file_paths = [
            oven_entity_train_trimmed_file_path,
            oven_query_train_trimmed_file_path,
        ]
        consolidated = consolidate_data(file_paths, candidate_pool_dict)
        # Note: right now we trim two files together, so each file may lose some entity_ids
        trimmed = trim_consolidated_data(consolidated, 25, 137)
        save_trimmed_data(trimmed, file_paths, trimmed_file_paths, candidate_pool_dict)

    # Convert OVEN data to MBEIR format
    if args.enable_mbeir_conversion:
        trimmed_file_paths = [
            oven_entity_train_trimmed_file_path,
            oven_query_train_trimmed_file_path,
        ]
        print("Converting OVEN data to MBEIR format...")
        print(f"Loading OVEN training data from {trimmed_file_paths}...")
        data_set_dict = {
            "train": trimmed_file_paths,
            "val": [oven_entity_val_file_path, oven_query_val_file_path],
        }
        for split, data_paths in data_set_dict.items():
            mbeir_format_oven_data_path = os.path.join(oven_dir, f"mbeir_oven_{split}.jsonl")

            oven_data = []
            for data_path in data_paths:
                oven_data.extend(load_jsonl_as_list(data_path))

            mbeir_entries = parallel_oven_to_mbeir(
                oven_data,
                oven_candidate_pool_6m_path,
                args.mbeir_data_dir,
                include_src_content=True,
            )
            save_list_as_jsonl(mbeir_entries, mbeir_format_oven_data_path, mode="w")

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_oven_data_path)
            print(f"MBEIR format OVEN {split} data saved to {mbeir_format_oven_data_path}")
            print(f"Total number of entries in {mbeir_format_oven_data_path}: {total_entries}")
            oven_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                oven_candidate_pool_6m_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, oven_cand_pool_dict)

    # Trim the number of candidates in the candidate pool
    if args.trim_candidate_pool:
        print("Trim 6M candidate pool to 1M candidates")
        oven_data = []
        for split in ["train", "val"]:
            data_path = os.path.join(oven_dir, f"mbeir_oven_{split}.jsonl")
            oven_data.extend(load_jsonl_as_list(data_path))

        skip_pool_set = set()
        for oven_entry in oven_data:
            query_src_content = json.loads(oven_entry["query_src_content"])
            skip_pool_set.add(query_src_content["entity_id"])
        print(f"{len(skip_pool_set)} candidates to skip")

        # Load the 6M candidate pool
        oven_wiki_6m_mbeir_format_pool = load_jsonl_as_list(oven_candidate_pool_6m_path)
        oven_wiki_6m_cand_pool_without_skip_set = []
        oven_wiki_6m_cand_pool_skip_set = []
        for entry in oven_wiki_6m_mbeir_format_pool:
            src_content = json.loads(entry["src_content"])
            if src_content["wikidata_id"] not in skip_pool_set:
                oven_wiki_6m_cand_pool_without_skip_set.append(entry)
            else:
                oven_wiki_6m_cand_pool_skip_set.append(entry)

        # Random sample 1M candidates
        random.shuffle(oven_wiki_6m_cand_pool_without_skip_set)
        augment_size = 1000000
        oven_wiki_6m_cand_pool_without_skip_set = oven_wiki_6m_cand_pool_without_skip_set[:augment_size]

        # Reassign document ids
        oven_wiki_1m_cand_pool = oven_wiki_6m_cand_pool_skip_set + oven_wiki_6m_cand_pool_without_skip_set
        document_id_start = 1
        for i, entry in enumerate(oven_wiki_1m_cand_pool):
            entry["did"] = f"{OVEN_DATASET_ID}:{document_id_start + i}"
        save_list_as_jsonl(oven_wiki_1m_cand_pool, oven_candidate_pool_1m_path, mode="w")
        print_mbeir_format_cand_pool_stats(oven_candidate_pool_1m_path)

        # Reassign dids in the data
        oven_wiki_1m_cand_dict = load_mbeir_format_oven_pool_file_as_dict(oven_candidate_pool_1m_path)
        for split in ["train", "val"]:
            mbeir_format_oven_data_path = os.path.join(oven_dir, f"mbeir_oven_{split}.jsonl")
            oven_data = load_jsonl_as_list(mbeir_format_oven_data_path)
            for oven_entry in oven_data:
                query_src_content = json.loads(oven_entry["query_src_content"])
                entity_id = query_src_content["entity_id"]
                oven_entry["pos_cand_list"] = [oven_wiki_1m_cand_dict[entity_id]]

            save_list_as_jsonl(oven_data, mbeir_format_oven_data_path, mode="w")

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_oven_data_path)
            print(f"MBEIR format OVEN {split} data saved to {mbeir_format_oven_data_path}")
            print(f"Total number of entries in {mbeir_format_oven_data_path}: {total_entries}")
            oven_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                oven_candidate_pool_1m_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, oven_cand_pool_dict)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        oven_train_candidate_pool_path = os.path.join(oven_dir, "mbeir_oven_train_cand_pool.jsonl")
        mbeir_format_oven_train_data_path = os.path.join(oven_dir, f"mbeir_oven_train.jsonl")
        assert os.path.exists(
            mbeir_format_oven_train_data_path
        ), f"File {mbeir_format_oven_train_data_path} does not exist"

        # Load the training data
        oven_train_candidate_pool = {}
        oven_cand_pool = load_mbeir_format_pool_file_as_dict(
            oven_candidate_pool_1m_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_oven_train_data = load_jsonl_as_list(mbeir_format_oven_train_data_path)
        for entry in mbeir_format_oven_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = oven_cand_pool[did]
                if did not in oven_train_candidate_pool:
                    oven_train_candidate_pool[did] = cand
                else:
                    if oven_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {oven_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        oven_train_candidate_pool_list = list(oven_train_candidate_pool.values())
        oven_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(oven_train_candidate_pool_list, oven_train_candidate_pool_path)
        print(f"Saved training candidate pool to {oven_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(oven_train_candidate_pool_path)

    # Assign infoseek did to the oven data
    if args.assign_did_from_infoseek_cand_pool:
        print("Assigning positive candidates from Infoseek candidate pool to OVEN queries...")
        infoseek_cand_pool_file_path = os.path.join(oven_dir, "mbeir_infoseek_cand_pool.jsonl")
        infoseek_train_candidate_pool_path = os.path.join(oven_dir, "mbeir_infoseek_train_cand_pool.jsonl")

        def load_wiki_pool_file_as_wikidata_id_to_did_dict(pool_file_path):
            """
            Load the candidate pool file into a dictionary.
            {doc_key : did} or {doc_key : entry}
            """
            pool_dict = {}
            assert pool_file_path.endswith(".jsonl"), "Only JSONL files are supported."

            with open(pool_file_path, "r") as f:
                for line in f:
                    entry = json.loads(line.strip())
                    src_content = json.loads(entry["src_content"])
                    doc_key = src_content["wikidata_id"]
                    if doc_key not in pool_dict:
                        pool_dict[doc_key] = []
                    did = entry["did"]
                    if did not in pool_dict[doc_key]:
                        pool_dict[doc_key].append(did)
            return pool_dict

        oven_1m_cand_dict = load_mbeir_format_pool_file_as_dict(
            oven_candidate_pool_1m_path, doc_key_to_content=True, key_type="did"
        )
        for split, infoseek_cand_pool_path in [
            ("train", infoseek_train_candidate_pool_path),
            ("val", infoseek_cand_pool_file_path),
        ]:
            infoseek_cand_pool = load_wiki_pool_file_as_wikidata_id_to_did_dict(infoseek_cand_pool_path)
            mbeir_format_oven_data_path = os.path.join(oven_dir, f"mbeir_oven_{split}.jsonl")
            oven_data = load_jsonl_as_list(mbeir_format_oven_data_path)
            for oven_entry in oven_data:
                infoseek_dids = []
                assert len(oven_entry["pos_cand_list"]) == 1, "Each oven entry should have only one positive candidate"
                did = oven_entry["pos_cand_list"][0]
                oven_cand = oven_1m_cand_dict[did]
                src_content = json.loads(oven_cand["src_content"])
                wikidata_id = src_content["wikidata_id"]
                infoseek_cand_did_list = infoseek_cand_pool.get(wikidata_id, None)
                if infoseek_cand_did_list:
                    for infoseek_did in infoseek_cand_did_list:
                        if infoseek_did not in infoseek_dids:
                            infoseek_dids.append(infoseek_did)
                oven_entry["pos_cand_list"].extend(infoseek_dids)

            mbeir_format_oven_data_merged_path = os.path.join(oven_dir, f"mbeir_oven_{split}_merged.jsonl")
            save_list_as_jsonl(oven_data, mbeir_format_oven_data_merged_path, mode="w")

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_oven_data_merged_path)
            print(f"MBEIR format OVEN merged {split} data saved to {mbeir_format_oven_data_merged_path}")
            print(f"Total number of entries in {mbeir_format_oven_data_merged_path}: {total_entries}")

            # Build combined pool
            infoseek_cand_pool = load_mbeir_format_pool_file_as_dict(
                infoseek_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            combined_pool_dict = {**infoseek_cand_pool, **oven_1m_cand_dict}
            print_mbeir_format_dataset_stats(data, combined_pool_dict)

    # Split val set into val and test
    if args.split_val_into_val_and_test:
        print("Split the OVEN validation set into validation and test sets")
        mbeir_oven_val_data_path = os.path.join(oven_dir, "mbeir_oven_val_merged.jsonl")
        mbeir_oven_val_data = load_jsonl_as_list(mbeir_oven_val_data_path)
        random.seed(2023)
        random.shuffle(mbeir_oven_val_data)
        new_oven_val_data = mbeir_oven_val_data[: len(mbeir_oven_val_data) // 2]
        new_oven_test_data = mbeir_oven_val_data[len(mbeir_oven_val_data) // 2 :]
        mbeir_oven_new_val_data_path = os.path.join(oven_dir, "mbeir_oven_new_val.jsonl")
        mbeir_oven_new_test_data_path = os.path.join(oven_dir, "mbeir_oven_new_test.jsonl")
        oven_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            oven_candidate_pool_1m_path, doc_key_to_content=True, key_type="did"
        )
        infoseek_cand_pool_file_path = os.path.join(oven_dir, "mbeir_infoseek_cand_pool.jsonl")
        infoseek_cand_pool = load_mbeir_format_pool_file_as_dict(
            infoseek_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        combined_pool_dict = {**infoseek_cand_pool, **oven_cand_pool_dict}
        print(f"Saved new validation data to {mbeir_oven_new_val_data_path}")
        save_list_as_jsonl(new_oven_val_data, mbeir_oven_new_val_data_path, mode="w")
        print_mbeir_format_dataset_stats(new_oven_val_data, combined_pool_dict)
        print(f"Saved new test data to {mbeir_oven_new_test_data_path}")
        save_list_as_jsonl(new_oven_test_data, mbeir_oven_new_test_data_path, mode="w")
        print_mbeir_format_dataset_stats(new_oven_test_data, combined_pool_dict)

    # Split candidate pool by task
    # Split the cand pool according to task
    if args.split_candidate_pool_by_task:
        print("Split the candidate pool according to task")

        # Load the candidate pool
        oven_cand_pool = load_jsonl_as_list(oven_candidate_pool_1m_path)

        # Split the candidate pool
        oven_task6_cand_pool = []
        oven_task8_cand_pool = []
        for oven_cand in oven_cand_pool:
            if oven_cand["modality"] == "text":
                oven_task6_cand_pool.append(oven_cand)
            elif oven_cand["modality"] == "image,text":
                oven_task8_cand_pool.append(oven_cand)
            else:
                raise ValueError(f"Unknown modality: {oven_cand['modality']}")
        print(f"Number of candidates for task 6: {len(oven_task6_cand_pool)}")
        print(f"Number of candidates for task 8: {len(oven_task8_cand_pool)}")

        # Save the candidate pool
        oven_task6_cand_pool_path = os.path.join(oven_dir, "mbeir_oven_task6_cand_pool.jsonl")
        oven_task8_cand_pool_path = os.path.join(oven_dir, "mbeir_oven_task8_cand_pool.jsonl")
        save_list_as_jsonl(oven_task6_cand_pool, oven_task6_cand_pool_path)
        save_list_as_jsonl(oven_task8_cand_pool, oven_task8_cand_pool_path)
        print(f"Saved task 6 candidate pool to {oven_task6_cand_pool_path}")
        print(f"Saved task 8 candidate pool to {oven_task8_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(oven_task6_cand_pool_path)
        print_mbeir_format_cand_pool_stats(oven_task8_cand_pool_path)

    # Split the query data according to task
    if args.split_query_data_by_task:
        print("Split the query data according to task")
        oven_1m_cand_dict = load_mbeir_format_pool_file_as_dict(
            oven_candidate_pool_1m_path, doc_key_to_content=True, key_type="did"
        )

        for split in ["val", "test"]:
            data_path = os.path.join(oven_dir, f"mbeir_oven_new_{split}.jsonl")
            task6_data_path = os.path.join(oven_dir, f"mbeir_oven_task6_{split}.jsonl")
            task8_data_path = os.path.join(oven_dir, f"mbeir_oven_task8_{split}.jsonl")

            # Load the data
            oven_data = load_jsonl_as_list(data_path)
            task6_data = []
            task8_data = []
            for entry in oven_data:
                pos_cand_did = entry["pos_cand_list"][0]
                pos_cand_modality = oven_1m_cand_dict[pos_cand_did]["modality"]
                if pos_cand_modality == "text":
                    task6_data.append(entry)
                elif pos_cand_modality == "image,text":
                    task8_data.append(entry)
                else:
                    raise ValueError(f"Unknown modality: {entry['query_modality']}")

            # Save the data
            save_list_as_jsonl(task6_data, task6_data_path)
            save_list_as_jsonl(task8_data, task8_data_path)

            infoseek_cand_pool_path = os.path.join(oven_dir, "mbeir_infoseek_cand_pool.jsonl")
            infoseek_cand_pool = load_mbeir_format_pool_file_as_dict(
                infoseek_cand_pool_path, doc_key_to_content=True, key_type="did"
            )
            combined_pool_dict = {**infoseek_cand_pool, **oven_1m_cand_dict}
            print(f"Saved task 6 data to {task6_data_path}")
            print_mbeir_format_dataset_stats(task6_data, combined_pool_dict)
            print(f"Saved task 8 data to {task8_data_path}")
            print_mbeir_format_dataset_stats(task8_data, combined_pool_dict)


if __name__ == "__main__":
    main()

import os
from PIL import Image
import PIL
import argparse
import json

from multiprocessing import Pool, cpu_count

from utils import (
    resize_and_convert_image_to_jpg,
    is_valid_image,
    get_dataset_id,
    format_string,
    count_entries_in_file,
    load_mbeir_format_pool_file_as_dict,
    generate_mbeir_format_doc_key,
    print_mbeir_format_cand_pool_stats,
    save_list_as_jsonl,
    load_jsonl_as_list,
    print_mbeir_format_dataset_stats,
    aggregate_candidates_for_mbeir_format_dataset,
)

EDIS_QUERY_MODALITY = "text"
EDIS_CANDIDATE_MODALITY = "image,text"
EDIS_DATASET_ID = get_dataset_id("EDIS")
assert EDIS_DATASET_ID is not None, "Unknown dataset name!"


def edis_to_mbeir_entry(
    edis_entry,
    candidate_pool,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Convert EDIS data format to MBEIR format.
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

    def create_mbeir_format_edis_candidate(candidate, modality_type):
        if modality_type == "image,text":
            img_path = os.path.join("mbeir_images", "edis_images", candidate["image"])
            candidate_txt = format_string(candidate["headline"])
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

        # Check if the image is valid and the text is not empty string
        if is_valid_image(os.path.join(mbeir_data_dir, img_path)) and candidate_txt:
            mbeir_candidate = {
                "did": None,
                "txt": candidate_txt,
                "img_path": img_path,
                "modality": modality_type,
            }
            # Generate document ID
            doc_key = generate_mbeir_format_doc_key(mbeir_candidate)
            did = candidate_pool.get(doc_key)
            assert did is not None, f"Document ID not found for doc_key: {doc_key}"
            return did
        else:
            return None

    dataset_id = get_dataset_id("EDIS")
    assert dataset_id is not None, "Unknown dataset name!"

    query_txt = format_string(edis_entry["query"])
    if not query_txt:  # Skip invalid queries
        return None

    mbeir_entry = {
        "qid": None,
        "query_txt": query_txt,
        "query_img_path": None,
        "query_modality": EDIS_QUERY_MODALITY,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    if include_src_content:
        query_src_content = {
            "id": str(edis_entry["id"]),
            "pos_cand_candidate_id_scores": [],
            "neg_cand_candidate_id_scores": [],
        }

    # Iterate through the candidates in the EDIS entry
    for candidate in edis_entry["candidates"]:
        new_candidate_did = create_mbeir_format_edis_candidate(candidate, "image,text")
        if not new_candidate_did:  # Skip invalid candidates
            continue
        if candidate["score"] == 3:
            mbeir_entry["pos_cand_list"].append(new_candidate_did)

            if include_src_content:
                query_src_content["pos_cand_candidate_id_scores"].append(
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "score": str(candidate["score"]),
                    }
                )

        else:  # for score 1 and 2
            mbeir_entry["neg_cand_list"].append(new_candidate_did)

            if include_src_content:
                query_src_content["neg_cand_candidate_id_scores"].append(
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "score": str(candidate["score"]),
                    }
                )

    if include_src_content:  # Cast to string to avoid JSON serialization error
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)

    # Check if we have any positive candidates
    if not mbeir_entry["pos_cand_list"]:
        print(f"Warning: No positive candidates for query: {mbeir_entry['query_txt']}")
        return None

    # Check if we have any negative candidates
    # if not mbeir_entry["neg_cand_list"]:
    #     print(f"Warning: No negative candidates for query: {mbeir_entry['query_txt']}")

    return mbeir_entry


def edis_to_mbeir(edis_data, candidate_pool_file_path, mbeir_data_dir, include_src_content=True):
    mbeir_entries = []
    # Load candidate pool
    cand_pool_dict = load_mbeir_format_pool_file_as_dict(
        candidate_pool_file_path,
        doc_key_to_content=False,
        key_type="mbeir_converted_key",
    )

    for edis_entry in edis_data:
        mbeir_entry = edis_to_mbeir_entry(
            edis_entry,
            cand_pool_dict,
            mbeir_data_dir,
            include_src_content=include_src_content,
        )
        if mbeir_entry:  # Skip invalid entries
            mbeir_entries.append(mbeir_entry)
    return mbeir_entries


def generate_edis_candidate_pool(
    data_set_list,
    edis_candidate_full_file_path,
    edis_candidate_pool_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate EDIS candidate pool in mbeir format and save it to a jsonl file.
    Here is the format of expected candidates_full.json
    [
    {
        "id":0,
        "contents":"In New York City, Hundreds Become U.S. Citizens Just in Time to Vote",
        "image":"00000000_04639_01.jpg"
    },...]
    """
    document_id = 1  # Note: We start from 1 for document IDs
    seen_candidates = set()  # To store headline and image path pairs

    with open(edis_candidate_pool_path, "w") as outfile:
        # EDIS' candidates_full.json missing some candidates,
        # So we need to generate the candidate pool from the train, dev, and test files
        for split, data_path in data_set_list:
            with open(data_path, "r") as source:
                edis_data = json.load(source)
                for edis_entry in edis_data:
                    for candidate in edis_entry["candidates"]:
                        # Note: we always store relative paths to MBEIR data directory
                        img_path = os.path.join("mbeir_images", "edis_images", candidate["image"])
                        headline = format_string(candidate["headline"])

                        # Skip invalid entries
                        if not headline:
                            print(f"Warning: Empty headline for image: {candidate}")
                            continue

                        # Skip invalid images
                        if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                            print(f"Warning: Invalid image: {candidate}")
                            continue

                        # Track if we've seen this candidate before
                        seen = (headline, img_path) in seen_candidates
                        if not seen:
                            candidate_pool_entry = {
                                "txt": headline,
                                "img_path": img_path,
                                "modality": EDIS_CANDIDATE_MODALITY,
                                "did": f"{EDIS_DATASET_ID}:{document_id}",
                            }
                            if include_src_content:
                                src_content = {
                                    "candidate_id": str(candidate["candidate_id"]),
                                }  # Cast to string to avoid JSON serialization error
                                candidate_pool_entry["src_content"] = json.dumps(src_content)
                            document_id += 1
                            outfile.write(json.dumps(candidate_pool_entry) + "\n")
                            seen_candidates.add((headline, img_path))

        # edis_candidate_full_file_path is a json file candidates_full.json
        assert edis_candidate_full_file_path.endswith(".json"), "Only JSON files are supported."
        with open(edis_candidate_full_file_path, "r") as source:
            edis_data = json.load(source)
            for edis_entry in edis_data:
                # Note: we always store relative paths to MBEIR data directory
                img_path = os.path.join("mbeir_images", "edis_images", edis_entry["image"])
                headline = format_string(edis_entry["contents"])

                # Skip invalid entries
                if not headline:
                    print(f"Warning: Empty headline for image: {edis_entry}")
                    continue

                # Skip invalid images
                if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                    print(f"Warning: Invalid image: {edis_entry}")
                    continue

                # Track if we've seen this candidate before
                seen = (headline, img_path) in seen_candidates
                if not seen:
                    candidate_pool_entry = {
                        "txt": headline,
                        "img_path": img_path,
                        "modality": EDIS_CANDIDATE_MODALITY,
                        "did": f"{EDIS_DATASET_ID}:{document_id}",
                    }
                    if include_src_content:
                        src_content = {
                            "id": str(edis_entry["id"]),
                        }  # Cast to string to avoid JSON serialization error
                        candidate_pool_entry["src_content"] = json.dumps(src_content)
                    document_id += 1  # increment for next entry
                    outfile.write(json.dumps(candidate_pool_entry) + "\n")
                    seen_candidates.add((headline, img_path))


def parallel_process_image_file(image_path):
    success = resize_and_convert_image_to_jpg(image_path)
    if not success:
        return 1
    return 0


def parallel_process_edis_image_directory(edis_images_dir):
    """
    Resize and convert all images in the given directory to JPG format.
    This function will delete corrupt images.
    Multiple processes are used to speed up the process.

    Here is the expected EDIS directory structure:
    ├── edis_images_dir
    │   ├── 00062500_04192_00.jpg
    │   ├── 00270684_40000_05.jpg
    """
    all_image_paths = []
    for root, _, files in os.walk(edis_images_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                all_image_paths.append(os.path.join(root, file))

    # Create a Pool of workers, defaulting to one per CPU core
    with Pool(cpu_count()) as p:
        results = p.map(parallel_process_image_file, all_image_paths)
        corrupted_files_count = sum(results)
    print(f"Total corrupted files: {corrupted_files_count}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format EDIS images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--edis_images_dir",
        type=str,
        default="mbeir_images/edis_images/",
        help="Relative directory path to save EDIS images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--edis_dir",
        type=str,
        default="src_data/edis/data",
        help="Relative directory path of EDIS files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Filter out corrupt images. 2. Resize images. 3. Convert images to JPEG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating EDIS candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting EDIS data to MBEIR format.",
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
    # Note: we keep the original project structure as in the EDIS dataset
    # So all the paths are hardcoded.
    edis_dir = os.path.join(args.mbeir_data_dir, args.edis_dir)
    edis_images_dir = os.path.join(args.mbeir_data_dir, args.edis_images_dir)
    edis_train_data_path = os.path.join(edis_dir, "train.json")
    edis_valid_data_path = os.path.join(edis_dir, "dev.json")
    edis_test_data_path = os.path.join(edis_dir, "test.json")
    edis_candidate_full_file_path = os.path.join(edis_dir, "candidates_full.json")
    edis_candidate_pool_path = os.path.join(edis_dir, "mbeir_edis_cand_pool.jsonl")

    # Process images
    if args.enable_image_processing:
        print(f"Processing images in {edis_images_dir}")
        parallel_process_edis_image_directory(edis_images_dir)

    # Generate candidate pool
    if args.enable_candidate_pool:
        print("Generating candidate pool in mbeir format...")
        data_set_list = [
            ("train", edis_train_data_path),
            ("val", edis_valid_data_path),
            ("test", edis_test_data_path),
        ]
        generate_edis_candidate_pool(
            data_set_list,
            edis_candidate_full_file_path,
            edis_candidate_pool_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"Candidate pool saved to {edis_candidate_pool_path}")
        # Print statistics
        print_mbeir_format_cand_pool_stats(edis_candidate_pool_path)

    # Convert EDIS data to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting EDIS data to MBEIR format...")
        data_set_list = [
            ("train", edis_train_data_path),
            ("val", edis_valid_data_path),
            ("test", edis_test_data_path),
        ]
        for split, data_path in data_set_list:
            mbeir_format_edis_data_path = os.path.join(edis_dir, f"mbeir_edis_{split}.jsonl")

            with open(data_path, "r") as file:
                data = json.load(file)

            mbeir_entries = edis_to_mbeir(
                data,
                edis_candidate_pool_path,
                args.mbeir_data_dir,
                include_src_content=True,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{EDIS_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_edis_data_path)

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_edis_data_path)
            print(f"MBEIR format EDIS {split} data saved to {mbeir_format_edis_data_path}")
            print(f"Total number of entries in {mbeir_format_edis_data_path}: {total_entries}")
            edis_cand_pool = load_mbeir_format_pool_file_as_dict(
                edis_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, edis_cand_pool)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        edis_train_candidate_pool_path = os.path.join(edis_dir, "mbeir_edis_train_cand_pool.jsonl")
        mbeir_format_edis_train_data_path = os.path.join(edis_dir, f"mbeir_edis_train.jsonl")
        assert os.path.exists(
            mbeir_format_edis_train_data_path
        ), f"File {mbeir_format_edis_train_data_path} does not exist"

        # Load the training data
        edis_train_candidate_pool = {}
        edis_cand_pool = load_mbeir_format_pool_file_as_dict(
            edis_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_edis_train_data = load_jsonl_as_list(mbeir_format_edis_train_data_path)
        for entry in mbeir_format_edis_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = edis_cand_pool[did]
                if did not in edis_train_candidate_pool:
                    edis_train_candidate_pool[did] = cand
                else:
                    if edis_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {edis_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        edis_train_candidate_pool_list = list(edis_train_candidate_pool.values())
        edis_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(edis_train_candidate_pool_list, edis_train_candidate_pool_path)
        print(f"Saved training candidate pool to {edis_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(edis_train_candidate_pool_path)


if __name__ == "__main__":
    main()

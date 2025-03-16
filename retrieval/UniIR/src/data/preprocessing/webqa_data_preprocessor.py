import argparse
import json
import os
import base64
from PIL import Image
from io import BytesIO
from multiprocessing import Pool, cpu_count
from functools import partial
import random

from utils import (
    resize_and_convert_image_to_jpg,
    is_valid_image,
    get_dataset_id,
    format_string,
    count_entries_in_file,
    count_mbeir_format_pool_entries_based_on_modality,
    load_mbeir_format_pool_file_as_dict,
    generate_mbeir_format_doc_key,
    check_duplicates_in_mbeir_format_cand_pool,
    print_mbeir_format_cand_pool_stats,
    save_list_as_jsonl,
    load_jsonl_as_list,
    print_mbeir_format_dataset_stats,
    aggregate_candidates_for_mbeir_format_dataset,
)

WEBQA_QUERY_MODALITY = "text"
WEBQA_DATASET_ID = get_dataset_id("WebQA")
assert WEBQA_DATASET_ID is not None, "Unknown dataset name!"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Decode WebQA images and refactor dataset to MBEIR format.")
    parser.add_argument(
        "--mbeir_data_dir",
        type=str,
        default="/data/UniIR/mbeir_data/",
        help="Absolute directory path of the MBEIR dataset.",
    )
    parser.add_argument(
        "--webqa_images_dir",
        type=str,
        default="mbeir_images/webqa_images/",
        help="Relative directory path to save WebQA images under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--webqa_dir",
        type=str,
        default="src_data/webqa/",
        help="Relative directory path of WebQA files folder under the MBEIR dataset directory.",
    )
    parser.add_argument(
        "--enable_image_processing",
        action="store_true",
        help="1. Decode images from base64. 2. Resize and convert images to JPG format.",
    )
    parser.add_argument(
        "--enable_candidate_pool",
        action="store_true",
        help="Enable generating candidate pool in mbeir format.",
    )
    parser.add_argument(
        "--enable_mbeir_conversion",
        action="store_true",
        help="Enable converting WebQA data to MBEIR format.",
    )
    parser.add_argument(
        "--enable_data_split",
        action="store_true",
        help="Enable splitting the WebQA train_val.json into train.json and test.json.",
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
        "--split_query_data_by_task",
        action="store_true",
        help="Enable splitting the query data according to task.",
    )
    return parser.parse_args()


def webqa_to_mbeir_entry(
    webqa_entry,
    candidate_pool,
    include_src_content=True,
):
    """Convert WebQA data format to MBEIR format.
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

    def create_webqa_candidate(fact, modality_type):
        if modality_type == "image,text":
            img_path = os.path.join("mbeir_images", "webqa_images", str(fact["image_id"]) + ".jpg")
            candidate_txt = format_string(fact["caption"])
        elif modality_type == "text":
            img_path = None
            candidate_txt = format_string(fact["fact"])
        else:
            raise ValueError(f"Unknown modality type: {modality_type}")

        candidate = {
            "did": None,
            "txt": candidate_txt,
            "img_path": img_path,
            "modality": modality_type,
        }

        # Generate document ID
        doc_key = generate_mbeir_format_doc_key(candidate)
        did = candidate_pool.get(doc_key, None)
        if did is None:
            print(f"Warning: Candidate not found in the candidate pool. doc_key: {doc_key}")
        return did

    query_txt = format_string(webqa_entry["Q"])
    if not query_txt:
        return None

    mbeir_entry = {
        "qid": None,
        "query_txt": query_txt,
        "query_img_path": None,
        "query_modality": WEBQA_QUERY_MODALITY,
        "query_src_content": None,
        "pos_cand_list": [],
        "neg_cand_list": [],
    }

    if include_src_content:
        query_src_content = {  # "A": str(webqa_entry.get("A", "")),
            "Guid": str(webqa_entry.get("Guid", "")),
            "Qcate": str(webqa_entry.get("Qcate", "")),
        }
        mbeir_entry["query_src_content"] = json.dumps(query_src_content)

    # Image-text candidates
    for fact in webqa_entry["img_posFacts"]:
        did = create_webqa_candidate(fact, "image,text")
        if did is not None:
            mbeir_entry["pos_cand_list"].append(did)
    for fact in webqa_entry["img_negFacts"]:
        did = create_webqa_candidate(fact, "image,text")
        if did is not None:
            mbeir_entry["neg_cand_list"].append(did)
    # Text-to-text candidates
    for fact in webqa_entry["txt_posFacts"]:
        did = create_webqa_candidate(fact, "text")
        if did is not None:
            mbeir_entry["pos_cand_list"].append(did)
    for fact in webqa_entry["txt_negFacts"]:
        did = create_webqa_candidate(fact, "text")
        if did is not None:
            mbeir_entry["neg_cand_list"].append(did)

    return mbeir_entry


def webqa_to_mbeir(webqa_data, candidate_pool_file_path, include_src_content=True):
    mbeir_entries = []

    # Load candidate pool
    candidate_pool = load_mbeir_format_pool_file_as_dict(candidate_pool_file_path, doc_key_to_content=False)

    for webqa_entry in webqa_data:
        mbeir_entry = webqa_to_mbeir_entry(
            webqa_entry,
            candidate_pool,
            include_src_content=include_src_content,
        )
        if mbeir_entry:
            mbeir_entries.append(mbeir_entry)
    return mbeir_entries


def generate_webqa_candidate_pool(
    webqa_train_val_json_path,
    webqa_test_json_path,
    webqa_candidate_pool_path,
    mbeir_data_dir,
    include_src_content=True,
):
    """
    Generate WebQA candidate pool in mbeir format.
    """

    def process_webqa_file(webqa_json_path, seen_texts, seen_image_text_pairs):
        cand_pool_entries = []

        with open(webqa_json_path, "r") as source:
            webqa_data = json.load(source)

            for entry_key, entry_value in webqa_data.items():
                all_fact_types = [
                    "img_negFacts",
                    "img_posFacts",
                    "txt_negFacts",
                    "txt_posFacts",
                    "img_Facts",
                    "txt_Facts",
                ]
                # WebQA_test.json only has 'img_Facts' and 'txt_Facts'

                for fact_type in all_fact_types:
                    # Check if the fact_type key exists in entry_value
                    if fact_type not in entry_value:
                        continue

                    for fact in entry_value[fact_type]:
                        if fact_type.startswith("img"):
                            img_id = fact["image_id"]
                            caption = format_string(fact["caption"])
                            img_path = os.path.join("mbeir_images", "webqa_images", str(img_id) + ".jpg")

                            # Check if the image-text pair has been seen before
                            if not caption:
                                continue
                            if (img_path, caption) in seen_image_text_pairs:
                                continue
                            if not is_valid_image(os.path.join(mbeir_data_dir, img_path)):
                                continue
                            candidate_pool_entry = {
                                "img_path": img_path,
                                "txt": caption,
                                "modality": "image,text",
                                "did": None,
                            }
                            if include_src_content:
                                src_content = {
                                    "image_id": str(fact.get("image_id", "")),
                                }
                                candidate_pool_entry["src_content"] = json.dumps(src_content)
                            cand_pool_entries.append(candidate_pool_entry)
                            seen_image_text_pairs.add((img_path, caption))

                        else:  # fact_type.startswith("txt")
                            txt = format_string(fact["fact"])
                            if not txt:
                                continue
                            if txt in seen_texts:
                                continue
                            candidate_pool_entry = {
                                "txt": txt,
                                "modality": "text",
                                "did": None,
                            }
                            if include_src_content:
                                src_content = {
                                    "snippet_id": str(fact.get("snippet_id", "")),
                                }
                                candidate_pool_entry["src_content"] = json.dumps(src_content)
                            cand_pool_entries.append(candidate_pool_entry)
                            seen_texts.add(txt)
        return cand_pool_entries

    seen_texts = set()
    seen_image_text_pairs = set()
    document_id = 1  # Note: We start from 1 for document IDs
    cand_pool_entries_merged = []
    cand_pool_entries_merged.extend(process_webqa_file(webqa_train_val_json_path, seen_texts, seen_image_text_pairs))
    cand_pool_entries_merged.extend(process_webqa_file(webqa_test_json_path, seen_texts, seen_image_text_pairs))
    for entry in cand_pool_entries_merged:
        entry["did"] = f"{WEBQA_DATASET_ID}:{document_id}"
        document_id += 1
    save_list_as_jsonl(cand_pool_entries_merged, webqa_candidate_pool_path)


def decode_and_save_base64_img(img_base64, save_path):
    image_data = BytesIO(base64.b64decode(img_base64))

    try:
        img = Image.open(image_data)

        # Convert palette images with transparency to RGBA
        if img.mode == "P":
            img = img.convert("RGBA")

        img = img.convert("RGB")  # Convert the image to RGB format
        img.save(save_path)
        return True
    except Exception as e:
        print(f"Failed to process {save_path}. Error: {e}")
        return False


def process_image(idx, webqa_images_dir, webqa_imgs_tsv_path):
    with open(webqa_imgs_tsv_path, "r") as fp:
        fp.seek(idx)
        imgid, img_base64 = fp.readline().strip().split("\t")

        # Construct the save path using imgid and webqa_images_dir
        save_path = os.path.join(webqa_images_dir, str(imgid) + ".jpg")
        success = decode_and_save_base64_img(img_base64, save_path)
        if not success:
            print(f"Failed to process imgid {imgid}, save_path: {save_path}")
            return 1
        # Now, resize and convert the saved image
        success = resize_and_convert_image_to_jpg(save_path)
        if not success:
            return 1  # Image was corrupted
    return 0


def main():
    args = parse_arguments()

    # Construct full paths
    # Note: we keep the original project structure as in the WebQA dataset
    # So all the paths are hardcoded.
    webqa_dir = os.path.join(args.mbeir_data_dir, args.webqa_dir)
    webqa_images_dir = os.path.join(args.mbeir_data_dir, args.webqa_images_dir)
    webqa_train_val_json_path = os.path.join(webqa_dir, "WebQA_train_val.json")
    webqa_test_json_path = os.path.join(webqa_dir, "WebQA_test.json")
    webqa_imgs_lineidx_path = os.path.join(webqa_dir, "imgs.lineidx")
    webqa_imgs_tsv_path = os.path.join(webqa_dir, "imgs.tsv")
    webqa_candidate_pool_path = os.path.join(webqa_dir, "mbeir_webqa_cand_pool.jsonl")

    if args.enable_image_processing:
        print("Decoding and saving images to webqa_images_dir...")
        # Create webqa image directory if it doesn't exist
        if not os.path.exists(webqa_images_dir):
            os.makedirs(webqa_images_dir)

        # Load lineidx
        with open(webqa_imgs_lineidx_path, "r") as fp_lineidx:
            lineidx = [int(i.strip()) for i in fp_lineidx.readlines()]
        print(f"Loaded {len(lineidx)} lineidx entries.")

        # Decode and save images to webqa_images_dir
        # Create a Pool of workers, defaulting to one per CPU core
        with Pool(cpu_count()) as p:
            partial_func = partial(
                process_image,
                webqa_images_dir=webqa_images_dir,
                webqa_imgs_tsv_path=webqa_imgs_tsv_path,
            )
            results = p.map(partial_func, lineidx)
            corrupted_files_count = sum(results)
        print(f"Total corrupted files: {corrupted_files_count}")

    if args.enable_candidate_pool:
        print("Generating candidate pool in mbeir format...")
        generate_webqa_candidate_pool(
            webqa_train_val_json_path,
            webqa_test_json_path,
            webqa_candidate_pool_path,
            args.mbeir_data_dir,
            include_src_content=True,
        )
        print(f"Candidate pool saved to {webqa_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(webqa_candidate_pool_path)

    # Convert WebQA dataset to MBEIR format
    if args.enable_mbeir_conversion:
        print("Converting WebQA dataset to MBEIR format...")
        with open(webqa_train_val_json_path, "r") as source:
            webqa_train_val_data = json.load(source)
            webqa_train_data = [entry for key, entry in webqa_train_val_data.items() if entry["split"] == "train"]
            webqa_val_data = [entry for key, entry in webqa_train_val_data.items() if entry["split"] == "val"]
        print(
            f"Loaded {len(webqa_train_data)} train entries and {len(webqa_val_data)} val entries from {webqa_train_val_json_path}."
        )
        print(f"Total number of entries in {webqa_train_val_json_path}: {len(webqa_train_val_data)}")
        data_split_list = ["train", "val"]
        for data_split in data_split_list:
            mbeir_format_webqa_path = os.path.join(webqa_dir, f"mbeir_webqa_{data_split}.jsonl")
            if data_split == "train":
                webqa_data = webqa_train_data
            elif data_split == "val":
                webqa_data = webqa_val_data
            else:
                raise ValueError(f"Unknown data split: {data_split}")
            mbeir_entries = webqa_to_mbeir(
                webqa_data,
                webqa_candidate_pool_path,
                include_src_content=True,
            )

            # Aggregate data
            mbeir_entries = aggregate_candidates_for_mbeir_format_dataset(mbeir_entries)

            # Generate query ID
            for i, entry in enumerate(mbeir_entries):
                entry.update({"qid": f"{WEBQA_DATASET_ID}:{i + 1}"})

            save_list_as_jsonl(mbeir_entries, mbeir_format_webqa_path)

            # Print statistics
            total_entries, data = count_entries_in_file(mbeir_format_webqa_path)
            print(f"MBEIR format WebQA data saved to {mbeir_format_webqa_path}")
            print(f"Total number of entries in {mbeir_format_webqa_path}: {total_entries}")
            webqa_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                webqa_candidate_pool_path, doc_key_to_content=True, key_type="did"
            )
            print_mbeir_format_dataset_stats(data, webqa_cand_pool_dict)

    # Split clean data into train and test sets
    # Note: Since WebQA didn't release their test set, we hold out 3500 entries from the train set as the validation set
    # and use the original validation set as the test set.
    if args.enable_data_split:
        print(
            "Splitting WebQA train_val.json into train.json and val.json and use the original val.json as test.json..."
        )
        train_test_data_path = os.path.join(webqa_dir, f"mbeir_webqa_train.jsonl")
        total_entries, data = count_entries_in_file(train_test_data_path)
        print(f"Total number of entries in {train_test_data_path}: {total_entries}")

        # Shuffle the data
        random.seed(2023)
        random.shuffle(data)

        # Define the number of entries for the val set
        num_val_entries = 3500

        # Split the data
        val_data = data[:num_val_entries]
        train_data = data[num_val_entries:]

        # Save the splits to different files
        train_data_after_split_path = os.path.join(webqa_dir, "mbeir_webqa_train_after_split.jsonl")
        val_data_after_split_path = os.path.join(webqa_dir, "mbeir_webqa_val_after_split.jsonl")
        test_data_after_split_path = os.path.join(webqa_dir, "mbeir_webqa_test_after_split.jsonl")
        save_list_as_jsonl(train_data, train_data_after_split_path)
        save_list_as_jsonl(val_data, val_data_after_split_path)
        val_data_before_split_path = os.path.join(webqa_dir, "mbeir_webqa_val.jsonl")
        test_data_after_split = load_jsonl_as_list(val_data_before_split_path)
        save_list_as_jsonl(test_data_after_split, test_data_after_split_path)

        webqa_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            webqa_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )

        total_train_entries, train_data = count_entries_in_file(train_data_after_split_path)
        print(f"Total number of entries in {train_data_after_split_path}: {total_train_entries}")
        print(f"Saved train data to {train_data_after_split_path}")
        print_mbeir_format_dataset_stats(train_data, webqa_cand_pool_dict)

        total_val_entries, val_data = count_entries_in_file(val_data_after_split_path)
        print(f"Total number of entries in {val_data_after_split_path}: {total_val_entries}")
        print(f"Saved val data to {val_data_after_split_path}")
        print_mbeir_format_dataset_stats(val_data, webqa_cand_pool_dict)

        total_test_entries, test_data = count_entries_in_file(test_data_after_split_path)
        print(f"Total number of entries in {test_data_after_split_path}: {total_test_entries}")
        print(f"Saved test data to {test_data_after_split_path}")
        print_mbeir_format_dataset_stats(test_data, webqa_cand_pool_dict)

    # Split the cand pool according to task
    if args.split_candidate_pool_by_task:
        print("Split the candidate pool according to task")
        # Load the candidate pool
        webqa_cand_pool = load_jsonl_as_list(webqa_candidate_pool_path)

        # Split the candidate pool according to task
        webqa_task1_cand_pool = []
        webqa_task2_cand_pool = []
        for webqa_cand in webqa_cand_pool:
            if webqa_cand["modality"] == "text":
                webqa_task1_cand_pool.append(webqa_cand)
            elif webqa_cand["modality"] == "image,text":
                webqa_task2_cand_pool.append(webqa_cand)
            else:
                raise ValueError(f"Unknown modality: {webqa_cand['modality']}")

        # Save the candidate pool
        webqa_task1_cand_pool_path = os.path.join(webqa_dir, "mbeir_webqa_task1_cand_pool.jsonl")
        webqa_task2_cand_pool_path = os.path.join(webqa_dir, "mbeir_webqa_task2_cand_pool.jsonl")
        save_list_as_jsonl(webqa_task1_cand_pool, webqa_task1_cand_pool_path)
        save_list_as_jsonl(webqa_task2_cand_pool, webqa_task2_cand_pool_path)
        print(f"Saved task 1 candidate pool to {webqa_task1_cand_pool_path}")
        print(f"Saved task 2 candidate pool to {webqa_task2_cand_pool_path}")
        print_mbeir_format_cand_pool_stats(webqa_task1_cand_pool_path)
        print_mbeir_format_cand_pool_stats(webqa_task2_cand_pool_path)

    # Split the query data according to task
    if args.split_query_data_by_task:
        print("Split the query data according to task")

        webq_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            webqa_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )

        for split in ["val", "test"]:
            data_path = os.path.join(webqa_dir, f"mbeir_webqa_{split}_after_split.jsonl")
            task1_data_path = os.path.join(webqa_dir, f"mbeir_webqa_task1_{split}.jsonl")
            task2_data_path = os.path.join(webqa_dir, f"mbeir_webqa_task2_{split}.jsonl")

            # Load the data
            webqa_data = load_jsonl_as_list(data_path)
            task1_data = []
            task2_data = []
            for entry in webqa_data:
                pos_cand_did = entry["pos_cand_list"][0]
                pos_cand_modality = webq_cand_pool_dict[pos_cand_did]["modality"]
                if pos_cand_modality == "text":
                    task1_data.append(entry)
                elif pos_cand_modality == "image,text":
                    task2_data.append(entry)
                else:
                    raise ValueError(f"Unknown modality: {entry['query_modality']}")

            # Save the data
            save_list_as_jsonl(task1_data, task1_data_path)
            save_list_as_jsonl(task2_data, task2_data_path)
            print(f"Saved task 1 data to {task1_data_path}")
            print_mbeir_format_dataset_stats(task1_data, webq_cand_pool_dict)
            print(f"Saved task 2 data to {task2_data_path}")
            print_mbeir_format_dataset_stats(task2_data, webq_cand_pool_dict)

    # Save the training candidate pool for hard negative mining
    if args.enable_training_candidate_pool:
        print("Generating training candidate pool in mbeir format...")
        webqa_train_candidate_pool_path = os.path.join(webqa_dir, "mbeir_webqa_train_cand_pool.jsonl")
        mbeir_format_webqa_train_data_path = os.path.join(webqa_dir, f"mbeir_webqa_train_after_split.jsonl")
        assert os.path.exists(
            mbeir_format_webqa_train_data_path
        ), f"File {mbeir_format_webqa_train_data_path} does not exist"

        # Load the training data
        webqa_train_candidate_pool = {}
        webqa_cand_pool = load_mbeir_format_pool_file_as_dict(
            webqa_candidate_pool_path, doc_key_to_content=True, key_type="did"
        )
        mbeir_format_webqa_train_data = load_jsonl_as_list(mbeir_format_webqa_train_data_path)
        for entry in mbeir_format_webqa_train_data:
            cand_list = entry["pos_cand_list"] + entry["neg_cand_list"]
            for did in cand_list:
                cand = webqa_cand_pool[did]
                if did not in webqa_train_candidate_pool:
                    webqa_train_candidate_pool[did] = cand
                else:
                    if webqa_train_candidate_pool[did] != cand:
                        print(f"Duplicate did for two candidates found: {webqa_train_candidate_pool[did]} and {cand}")

        # Save the training candidate pool
        webqa_train_candidate_pool_list = list(webqa_train_candidate_pool.values())
        webqa_train_candidate_pool_list.sort(key=lambda x: int(x["did"].split(":")[1]))
        save_list_as_jsonl(webqa_train_candidate_pool_list, webqa_train_candidate_pool_path)
        print(f"Saved training candidate pool to {webqa_train_candidate_pool_path}")
        print_mbeir_format_cand_pool_stats(webqa_train_candidate_pool_path)


if __name__ == "__main__":
    main()

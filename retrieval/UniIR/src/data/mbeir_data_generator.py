import json
import random
import argparse
import os

from preprocessing.utils import (
    save_list_as_jsonl,
    print_mbeir_format_dataset_stats,
    save_and_print_mbeir_format_dataset_stats,
    count_entries_in_file,
    load_jsonl_as_list,
    print_mbeir_format_cand_pool_stats,
    load_mbeir_format_pool_file_as_dict,
    DATASET_IDS,
    get_dataset_name,
    get_mbeir_task_id,
)

_100K = 100000
_50K = 50000
_15K = 15000


# Function to load and optionally upsample dataset
def load_and_upsample(file_path, target_size, enable_upsampling):
    with open(file_path, "r") as file:
        data = [json.loads(line) for line in file]

    original_size = len(data)

    if enable_upsampling and original_size != target_size:
        while len(data) < target_size:
            data.extend(random.choices(data, k=target_size - len(data)))

    return data, original_size


def unify_upsample_mbeir_data(data_dir, data_split, upsample, datasets_info, shuffle=True):
    # Load and concatenate
    print(f"Unify {data_split} data...")
    union_data = []
    for name, dataset_info in datasets_info.items():
        name = name.lower()
        if dataset_info["include"]:
            file_path = os.path.join(data_dir, f"mbeir_{name}_{data_split}.jsonl")
            if upsample:
                data, _ = load_and_upsample(file_path, dataset_info["target"], dataset_info["up_sampling"])
                print(
                    f"Dataset {name}, Perform Up-sampling: {dataset_info['up_sampling']}, "
                    f"Original Size: {dataset_info['original']}, New Size: {len(data)}"
                )
            else:
                data = load_jsonl_as_list(file_path)
                print(f"Dataset {name}, Perform Up-sampling: {dataset_info['up_sampling']}, Size: {len(data)}")
            union_data.extend(data)
        else:
            print(f"Dataset {name}, Not included in the final aggregated {data_split} dataset")

    if shuffle:
        # Shuffle the unified dataset
        random.shuffle(union_data)
    else:
        # Sort the unified dataset
        union_data.sort(key=lambda x: (int(x["qid"].split(":")[0]), int(x["qid"].split(":")[1])))

    return union_data


def unify_mbeir_cand_pool(
    cand_pool_dir,
    dataset_to_cand_pool_file_middle_name_map,
    datasets_info,
    shuffle=False,
):
    print(f"Unify all candidate pools from {cand_pool_dir}...")
    union_cand_pool = []
    for name, dataset_info in datasets_info.items():
        cand_pool_file_middle_names = dataset_to_cand_pool_file_middle_name_map[name]
        if dataset_info["include"]:
            for cand_pool_file_middle_name in cand_pool_file_middle_names:
                cand_pool_file_path = os.path.join(cand_pool_dir, f"mbeir_{cand_pool_file_middle_name}_cand_pool.jsonl")
                print(f"Loading candidate pool from {cand_pool_file_path}...")
                cand_pool = load_jsonl_as_list(cand_pool_file_path)
                union_cand_pool.extend(cand_pool)
            print(f"Aggregated dataset {name} candidate pool")
        else:
            print(f"Dataset {name} candidate pool, Not included in the final aggregated candidate pool")

    if shuffle:
        # Shuffle the unified candidate pool
        random.shuffle(union_cand_pool)
    else:
        # Sort the unified candidate pool
        union_cand_pool.sort(key=lambda x: (int(x["did"].split(":")[0]), int(x["did"].split(":")[1])))

    return union_cand_pool


def fetch_original_dataset_sizes(train_data_dir, datasets_info):
    for name in datasets_info.keys():
        orig_name = name
        name = name.lower()
        file_path = os.path.join(train_data_dir, f"mbeir_{name}_train.jsonl")
        with open(file_path, "r") as file:
            data = [json.loads(line) for line in file]
        datasets_info[orig_name]["original"] = len(data)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Manage mbeir_train datasets.")
    parser.add_argument(
        "--mbeir_data_dir",
        default="/data/UniIR/mbeir_data",
        help="Path to mbeir_data directory",
    )
    parser.add_argument(
        "--print_original_train_data_sizes",
        action="store_true",
        help="Print the original sizes of datasets without any operation",
    )
    parser.add_argument(
        "--generate_union_train_cand_pool",
        action="store_true",
        help="Generate union train candidate pool",
    )
    parser.add_argument(
        "--generate_union_test_cand_pool",
        action="store_true",
        help="Generate union test candidate pool",
    )
    parser.add_argument(
        "--generate_union_all_cand_pool",
        action="store_true",
        help="Generate union candidate pool for all the datasets and all the splits",
    )
    parser.add_argument(
        "--unify_train_data",
        action="store_true",
        help="Aggregate train datasets without performing upsampling",
    )
    parser.add_argument(
        "--unify_and_upsample_train_data",
        action="store_true",
        help="Aggregate train datasets and perform upsampling",
    )
    parser.add_argument(
        "--unify_and_upsample_train_data_with_hard_negs",
        action="store_true",
        help="Aggregate train datasets with hard negatives and perform upsampling",
    )
    parser.add_argument(
        "--hard_negs_dir_name",
        default="hard_negs_dir",
        help="Name of the directory containing hard negatives, should be HardNegs/experiment_name",
    )
    parser.add_argument(
        "--generate_qrels",
        action="store_true",
        help="Generate qrels file",
    )
    parser.add_argument(
        "--assign_task_ids",
        action="store_true",
        help="Assign task ids to datasets",
    )
    parser.add_argument(
        "--generate_union_val_data",
        action="store_true",
        help="Generate union validation data",
    )
    parser.add_argument(
        "--generate_held_n_dataset_out_data",
        action="store_true",
        help="Generate held n dataset out dataset",
    )
    parser.add_argument(
        "--generate_held_n_task_out_data",
        action="store_true",
        help="Generate held n task out dataset",
    )
    parser.add_argument(
        "--generate_held_1_domain_out_data",
        action="store_true",
        help="Generate held 1 out domain dataset",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Dictionary with dataset names, their desired final sizes, and whether to upsample them or not,
    # and whether to include them in the final aggregated dataset or not.
    datasets_info = {
        "VisualNews": {"target": _100K, "up_sampling": False, "include": True},
        "MSCOCO": {"target": _100K, "up_sampling": False, "include": True},
        "Fashion200K": {"target": _15K, "up_sampling": False, "include": True},
        "WebQA": {"target": _100K, "up_sampling": True, "include": True},
        "EDIS": {"target": _50K, "up_sampling": True, "include": True},
        "NIGHTS": {"target": _50K, "up_sampling": True, "include": True},
        "OVEN": {"target": _100K, "up_sampling": False, "include": True},
        "INFOSEEK": {"target": _100K, "up_sampling": False, "include": True},
        "FashionIQ": {"target": _50K, "up_sampling": True, "include": True},
        "CIRR": {"target": _50K, "up_sampling": True, "include": True},
    }

    # Construct the full path
    train_data_dir = os.path.join(args.mbeir_data_dir, "train")
    union_train_data_dir = os.path.join(train_data_dir, "union_train")

    cand_pool_dir = os.path.join(args.mbeir_data_dir, "cand_pool")
    train_cand_pool_dir = os.path.join(cand_pool_dir, "train_cand_pool")
    union_pool_dir = os.path.join(cand_pool_dir, "union_pool")

    # Fetch original sizes for datasets
    print("Fetching original sizes for datasets...")
    fetch_original_dataset_sizes(train_data_dir, datasets_info)

    if args.print_original_train_data_sizes:
        print("Original Sizes of Datasets:")
        for name, sizes in datasets_info.items():
            print(f"Dataset {name}: Size = {sizes['original']}")

    # You need to run them in order
    # Step 1.
    if args.generate_union_train_cand_pool:
        # Load and concatenate
        print("Unify train candidate pool...")
        union_train_cand_pool = []
        for name, dataset_info in datasets_info.items():
            name = name.lower()
            if dataset_info["include"]:
                file_path = os.path.join(train_cand_pool_dir, f"mbeir_{name}_train_cand_pool.jsonl")
                data = load_jsonl_as_list(file_path)
                union_train_cand_pool.extend(data)
                print(f"Unify dataset {name} train candidate pool")
            else:
                print(f"Dataset {name} train candidate pool, Not included in the final aggregated train candidate pool")

        # Sort the unified candidate pool
        union_train_cand_pool.sort(key=lambda x: (int(x["did"].split(":")[0]), int(x["did"].split(":")[1])))

        # Save to the new JSONL file
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_list_as_jsonl(union_train_cand_pool, union_train_cand_pool_file_path)
        print(f"Saved union training candidate pool to {union_train_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(union_train_cand_pool_file_path, print_duplicate=False)

    # Step 1.
    if args.generate_union_test_cand_pool:
        # Aggregate candidate pools
        print("Unify test candidate pools...")
        dataset_to_cand_pool_file_middle_name_map = {
            "VisualNews": ["visualnews_task0", "visualnews_task3"],
            "MSCOCO": [
                "mscoco_task0_test",
                "mscoco_task3_test",
            ],  # Hack for MSCOCO, union pool only contains data from MSCOCO test set
            "Fashion200K": ["fashion200k_task0", "fashion200k_task3"],
            "WebQA": ["webqa_task1", "webqa_task2"],
            "EDIS": ["edis_task2"],
            "NIGHTS": ["nights_task4"],
            "OVEN": ["oven_task6", "oven_task8"],
            "INFOSEEK": ["infoseek_task6", "infoseek_task8"],
            "FashionIQ": ["fashioniq_task7"],
            "CIRR": ["cirr_task7"],
        }
        union_cand_pool = unify_mbeir_cand_pool(
            cand_pool_dir=cand_pool_dir,
            dataset_to_cand_pool_file_middle_name_map=dataset_to_cand_pool_file_middle_name_map,
            datasets_info=datasets_info,
            shuffle=False,
        )
        union_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_test_cand_pool.jsonl")
        save_list_as_jsonl(union_cand_pool, union_cand_pool_file_path)
        print(f"Saved union candidate pool to {union_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(union_cand_pool_file_path, print_duplicate=False)

    # Step 1.1
    # The union all pool contains all the candidates from all the datasets from all the splits
    if args.generate_union_all_cand_pool:
        # Aggregate candidate pools
        print("Generate union all candidate pool...")
        dataset_to_cand_pool_file_middle_name_map = {
            "VisualNews": ["visualnews"],
            "MSCOCO": ["mscoco"],
            "Fashion200K": ["fashion200k"],
            "WebQA": ["webqa"],
            "EDIS": ["edis"],
            "NIGHTS": ["nights"],
            "OVEN": ["oven"],
            "INFOSEEK": ["infoseek"],
            "FashionIQ": ["fashioniq"],
            "CIRR": ["cirr"],
        }
        unsplit_cand_pool_dir = os.path.join(cand_pool_dir, "unsplit_cand_pool")
        union_cand_pool = unify_mbeir_cand_pool(
            cand_pool_dir=unsplit_cand_pool_dir,
            dataset_to_cand_pool_file_middle_name_map=dataset_to_cand_pool_file_middle_name_map,
            datasets_info=datasets_info,
            shuffle=False,
        )
        union_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_all_cand_pool.jsonl")
        save_list_as_jsonl(union_cand_pool, union_cand_pool_file_path)
        print(f"Saved union candidate pool to {union_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(union_cand_pool_file_path, print_duplicate=False)

    # Step 2.
    if args.unify_train_data:
        union_data = unify_upsample_mbeir_data(
            data_dir=train_data_dir,
            data_split="train",
            upsample=False,
            datasets_info=datasets_info,
            shuffle=True,
        )
        union_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_train.jsonl")
        # We are using train candidate pool since MSCOCO training and val set is not included in the union pool
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(union_data, union_train_file_path, union_train_cand_pool_file_path)

    if args.unify_and_upsample_train_data:
        union_data = unify_upsample_mbeir_data(
            data_dir=train_data_dir,
            data_split="train",
            upsample=True,
            datasets_info=datasets_info,
            shuffle=True,
        )
        union_up_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_up_train.jsonl")
        # We are using train candidate pool since MSCOCO training and val is not included in the union pool
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(union_data, union_up_train_file_path, union_train_cand_pool_file_path)

    # Step 3.
    if args.assign_task_ids:
        # Assign task ids to datasets
        print("Assigning task ids to datasets...")
        for split in ["train", "val", "test", os.path.join("train", "union_train")]:
            if "train" in split:
                cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
                cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                    cand_pool_file_path, doc_key_to_content=True, key_type="did"
                )
                print(f"Loading candidate pool from {cand_pool_file_path}...")
            data_dir = os.path.join(args.mbeir_data_dir, split)
            for data_file in os.listdir(data_dir):
                if data_file.endswith(".jsonl"):
                    print(f"\nProcessing {os.path.join(data_dir, data_file)}...")
                    if not "train" in split:
                        # Extract dataset_name and dataset_split from file_name
                        file_name_no_ext = os.path.splitext(data_file)[0]
                        parts = file_name_no_ext.split("_")
                        dataset_split = parts[-1]
                        middle_name = "_".join(parts[1:-1])  # e.g. oven_task6
                        cand_pool_file_path = os.path.join(cand_pool_dir, "mbeir_" + middle_name + "_cand_pool.jsonl")
                        # Hack for MSCOCO dataset
                        if "mscoco" in middle_name:
                            cand_pool_file_path = os.path.join(
                                cand_pool_dir,
                                f"mbeir_{middle_name}_{split}_cand_pool.jsonl",
                            )
                        print(f"Loading candidate pool from {cand_pool_file_path}...")
                        cand_pool_dict = load_mbeir_format_pool_file_as_dict(
                            cand_pool_file_path, doc_key_to_content=True, key_type="did"
                        )

                    # Assign task ids to data
                    data = load_jsonl_as_list(os.path.join(data_dir, data_file))
                    for entry in data:
                        query_modality = entry["query_modality"]
                        pos_cand_did = entry["pos_cand_list"][0]
                        pos_cand_modality = cand_pool_dict[pos_cand_did]["modality"]
                        task_id = get_mbeir_task_id(query_modality, pos_cand_modality)
                        entry["task_id"] = task_id
                    save_list_as_jsonl(data, os.path.join(data_dir, data_file))
                    print(f"Saved {os.path.join(data_dir, data_file)} with task ids")

                    # Print task ids counts
                    task_ids = [entry["task_id"] for entry in data]
                    task_ids_counts = {id: task_ids.count(id) for id in set(task_ids)}
                    print("Task ID Counts:")
                    for task, count in task_ids_counts.items():
                        print(f"Task ID: {task} - Count: {count}")

    # Step 5. Unify validation data
    if args.generate_union_val_data:
        dataset_to_val_data_file_middle_name_map = {
            "VisualNews": ["visualnews_task0", "visualnews_task3"],
            "MSCOCO": ["mscoco_task0", "mscoco_task3"],
            "Fashion200K": ["fashion200k_task0", "fashion200k_task3"],
            "WebQA": ["webqa_task1", "webqa_task2"],
            "EDIS": ["edis_task2"],
            "NIGHTS": ["nights_task4"],
            "OVEN": ["oven_task6", "oven_task8"],
            "INFOSEEK": ["infoseek_task6", "infoseek_task8"],
            "FashionIQ": ["fashioniq_task7"],
            "CIRR": ["cirr_task7"],
        }
        val_data_dir = os.path.join(args.mbeir_data_dir, "val")
        union_val_data = []
        for name, dataset_info in datasets_info.items():
            val_data_file_middle_names = dataset_to_val_data_file_middle_name_map[name]
            if dataset_info["include"]:
                for val_data_file_middle_name in val_data_file_middle_names:
                    val_data_file_path = os.path.join(val_data_dir, f"mbeir_{val_data_file_middle_name}_val.jsonl")
                    print(f"Loading validation data from {val_data_file_path}...")
                    val_data = load_jsonl_as_list(val_data_file_path)
                    union_val_data.extend(val_data)
                print(f"Aggregated dataset {name} validation data")
            else:
                print(f"Dataset {name} validation data, Not included in the final aggregated validation data")

        # Sort the unified validation data
        union_val_data.sort(key=lambda x: (int(x["qid"].split(":")[0]), int(x["qid"].split(":")[1])))

        # Save to the new JSONL file
        union_val_data_dir = os.path.join(args.mbeir_data_dir, "val", "union_val")
        os.makedirs(union_val_data_dir, exist_ok=True)
        union_val_data_file_path = os.path.join(union_val_data_dir, "mbeir_union_val.jsonl")
        save_list_as_jsonl(union_val_data, union_val_data_file_path)
        print(f"Saved union validation data to {union_val_data_file_path}")

        # Create union validation candidate pool
        dataset_to_val_cand_pool_file_middle_name_map = {
            "VisualNews": ["visualnews_task0", "visualnews_task3"],
            "MSCOCO": [
                "mscoco_task0_val",
                "mscoco_task3_val",
            ],  # Hack for MSCOCO, union pool only contains data from MSCOCO test set
            "Fashion200K": ["fashion200k_task0", "fashion200k_task3"],
            "WebQA": ["webqa_task1", "webqa_task2"],
            "EDIS": ["edis_task2"],
            "NIGHTS": ["nights_task4"],
            "OVEN": ["oven_task6", "oven_task8"],
            "INFOSEEK": ["infoseek_task6", "infoseek_task8"],
            "FashionIQ": ["fashioniq_task7"],
            "CIRR": ["cirr_task7"],
        }
        union_val_cand_pool = unify_mbeir_cand_pool(
            cand_pool_dir=cand_pool_dir,
            dataset_to_cand_pool_file_middle_name_map=dataset_to_val_cand_pool_file_middle_name_map,
            datasets_info=datasets_info,
            shuffle=False,
        )
        # Save to the new JSONL file
        union_pool_dir = os.path.join(cand_pool_dir, "union_pool")
        union_val_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_val_cand_pool.jsonl")
        save_list_as_jsonl(union_val_cand_pool, union_val_cand_pool_file_path)
        print(f"Saved union validation candidate pool to {union_val_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(union_val_cand_pool_file_path, print_duplicate=False)

        # Trim the union validation candidate pool to only contain candidates from the union validation data
        union_val_cand_pool_dict = load_mbeir_format_pool_file_as_dict(
            union_val_cand_pool_file_path, doc_key_to_content=True, key_type="did"
        )
        trimmed_union_val_cand_pool = {}
        for entry in union_val_data:
            for did in entry["pos_cand_list"]:
                trimmed_union_val_cand_pool[did] = union_val_cand_pool_dict[did]
            for did in entry["neg_cand_list"]:
                trimmed_union_val_cand_pool[did] = union_val_cand_pool_dict[did]

        # Save to the new JSONL file
        trimmed_union_val_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_val_cand_pool.jsonl")
        save_list_as_jsonl(
            list(trimmed_union_val_cand_pool.values()),
            trimmed_union_val_cand_pool_file_path,
        )
        print(f"Saved trimmed union validation candidate pool to {trimmed_union_val_cand_pool_file_path}")
        print_mbeir_format_cand_pool_stats(trimmed_union_val_cand_pool_file_path, print_duplicate=False)

        # Print unified validation data stats
        print_mbeir_format_dataset_stats(union_val_data, union_val_cand_pool_dict)

    # Step 4.
    if args.generate_qrels:
        # Generate qrels file
        print("Generating qrels file...")
        qrels_dir = os.path.join(args.mbeir_data_dir, "qrels")
        os.makedirs(qrels_dir, exist_ok=True)
        for split in ["train", "val", "test"]:
            data_dir = os.path.join(args.mbeir_data_dir, split)
            for data_file in os.listdir(data_dir):
                if data_file.endswith(".jsonl"):
                    file_name = os.path.basename(data_file)

                    # Extract dataset_name and current_split from file_name
                    file_name_no_ext = os.path.splitext(file_name)[0]
                    parts = file_name_no_ext.split("_")
                    dataset_split = parts[-1]
                    middle_name = "_".join(parts[1:-1])

                    qrels_file = os.path.join(qrels_dir, f"mbeir_{middle_name}_{dataset_split}_qrels.txt")
                    print(f"\nGenerating qrels file {qrels_file}...")

                    mbeir_data = load_jsonl_as_list(os.path.join(data_dir, file_name))
                    print(f"Loading data from {os.path.join(data_dir, file_name)}...")

                    with open(qrels_file, "w") as outfile:
                        for entry in mbeir_data:
                            qid = entry["qid"]
                            task_id = entry["task_id"]
                            # Write positive candidates with relevance score 1
                            for cand_id in entry["pos_cand_list"]:
                                outfile.write(f"{qid} 0 {cand_id} 1 {task_id}\n")
                    print(f"Generated qrels file {qrels_file}")

    # Exp 1 Held N dataset out
    if args.generate_held_n_dataset_out_data:
        # Load union upsampled train data
        union_up_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_up_train.jsonl")
        union_up_train_data = load_jsonl_as_list(union_up_train_file_path)
        held_in_data = []
        held_out_dataset_names = ["OVEN", "CIRR", "WebQA", "VisualNews", "Fashion200K"]
        for entry in union_up_train_data:
            if get_dataset_name(entry["qid"]) in held_out_dataset_names:
                continue
            held_in_data.append(entry)
        held_in_data_dir = os.path.join(train_data_dir, "EXP", "HeldNDataOut")
        os.makedirs(held_in_data_dir, exist_ok=True)
        held_in_data_file_path = os.path.join(held_in_data_dir, "mbeir_held_n_dataset_out_train.jsonl")
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(held_in_data, held_in_data_file_path, union_train_cand_pool_file_path)

    # Exp 2. Held N task out
    if args.generate_held_n_task_out_data:
        # Load union upsampled train data
        union_up_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_up_train.jsonl")
        union_up_train_data = load_jsonl_as_list(union_up_train_file_path)
        held_in_data = []
        held_out_task_id = [0, 2, 8]
        for entry in union_up_train_data:
            if entry["task_id"] in held_out_task_id:
                continue
            held_in_data.append(entry)
        held_in_data_dir = os.path.join(train_data_dir, "EXP", "HeldNTaskOut")
        os.makedirs(held_in_data_dir, exist_ok=True)
        held_in_data_file_path = os.path.join(held_in_data_dir, "mbeir_held_n_task_out_train.jsonl")
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(held_in_data, held_in_data_file_path, union_train_cand_pool_file_path)

    # Exp 3 Held 1 domain out
    if args.generate_held_1_domain_out_data:
        # Load union upsampled train data
        union_up_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_up_train.jsonl")
        union_up_train_data = load_jsonl_as_list(union_up_train_file_path)
        held_in_data = []
        held_out_domain = "news"
        held_out_dataset_names = ["EDIS", "VisualNews"]
        for entry in union_up_train_data:
            if get_dataset_name(entry["qid"]) in held_out_dataset_names:
                continue
            held_in_data.append(entry)
        held_in_data_dir = os.path.join(train_data_dir, "EXP", "Held1DomainOut")
        os.makedirs(held_in_data_dir, exist_ok=True)
        held_in_data_file_path = os.path.join(held_in_data_dir, "mbeir_held_1_domain_out_train.jsonl")
        union_train_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_train_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(held_in_data, held_in_data_file_path, union_train_cand_pool_file_path)

    # Hard negative mining
    if args.unify_and_upsample_train_data_with_hard_negs:
        hard_negs_dir = os.path.join(args.mbeir_data_dir, "train", args.hard_negs_dir_name)
        print(f"Unify and upsample train data with hard negatives from {hard_negs_dir}...")
        union_data = unify_upsample_mbeir_data(
            data_dir=hard_negs_dir,
            data_split="hard_negs_train",
            upsample=True,
            datasets_info=datasets_info,
            shuffle=False,
        )
        union_up_hard_negs_train_file_path = os.path.join(union_train_data_dir, "mbeir_union_up_hard_negs_train.jsonl")
        # We are using union_all candidate pool since MSCOCO has separate candidate pools for train and val and test.
        union_all_cand_pool_file_path = os.path.join(union_pool_dir, "mbeir_union_all_cand_pool.jsonl")
        save_and_print_mbeir_format_dataset_stats(
            union_data,
            union_up_hard_negs_train_file_path,
            union_all_cand_pool_file_path,
        )


if __name__ == "__main__":
    main()

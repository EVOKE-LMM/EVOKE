"""
Module for Updating YAML Configuration Files in MBEIR Experiments.
For example, updating the instruction status, see the provided scripts.
"""

import argparse
import yaml
from copy import deepcopy


def load_yaml(file_path):
    with open(file_path) as f:
        return yaml.safe_load(f)


def save_yaml(data, file_path):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, default_flow_style=False)


def print_yaml(data):
    print(yaml.dump(data, default_flow_style=False, sort_keys=False, indent=4))


def update_mbeir_yaml_instruct_status(yaml_file_path, enable_instruct):
    print(f"Updating YAML {yaml_file_path} for instruct status: {enable_instruct}")
    yaml_data = load_yaml(yaml_file_path)
    if enable_instruct:
        yaml_data["experiment"]["instruct_status"] = "Instruct"
        if "data_config" in yaml_data:
            yaml_data["data_config"]["enable_query_instruct"] = True
        else:
            print(f"YAML {yaml_file_path} does not have data_config.")
    else:
        yaml_data["experiment"]["instruct_status"] = "NoInstruct"
        if "data_config" in yaml_data:
            yaml_data["data_config"]["enable_query_instruct"] = False
        else:
            print(f"YAML {yaml_file_path} does not have data_config.")
    print(f"Updated YAML {yaml_file_path} for instruct status:{enable_instruct} :")
    print_yaml(yaml_data)
    save_yaml(yaml_data, yaml_file_path)


def update_mbeir_config_dir_instruct_status(config_dir, enable_instruct):
    print(f"Updating config dir {config_dir} for instruct status: {enable_instruct}")

    # Update the Embed YAML file
    embed_yaml_file_path = f"{config_dir}/embed.yaml"
    update_mbeir_yaml_instruct_status(embed_yaml_file_path, enable_instruct)

    # Update the Index YAML file
    index_yaml_file_path = f"{config_dir}/index.yaml"
    update_mbeir_yaml_instruct_status(index_yaml_file_path, enable_instruct)

    # Update the Retrieval YAML file
    retrieval_yaml_file_path = f"{config_dir}/retrieval.yaml"
    update_mbeir_yaml_instruct_status(retrieval_yaml_file_path, enable_instruct)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Updating experiment configurations.")
    parser.add_argument(
        "--update_mbeir_yaml_instruct_status",
        action="store_true",
        default=True,
        help="Update instruct status for mbeir yamls",
    )
    parser.add_argument("--mbeir_yaml_file_path", type=str, default="/home/du/pan1/new_knowledge/real_new/mmrag/UniIR/src/models/uniir_clip/clip_scorefusion/configs_scripts/large/eval/inbatch/retrieval.yaml")
    parser.add_argument(
        "--enable_instruct",
        # required=True,
        default="True",
        choices=["True", "False"],
        help="Whether to enable instruct or not. Required input.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # Update mbeir yaml instruct status
    enable_instruct = args.enable_instruct == "True"
    if args.update_mbeir_yaml_instruct_status:
        if args.mbeir_yaml_file_path == "ReplaceMe":
            print("The default YAML file path has not been replaced with an actual file path.")
        update_mbeir_yaml_instruct_status(args.mbeir_yaml_file_path, enable_instruct)

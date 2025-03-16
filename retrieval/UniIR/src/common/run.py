import os

UNIIR_DIR="/home/du/pan1/new_knowledge/real_new/mmrag/UniIR" # <--- Change this to the UniIR directory
MBEIR_DATA_DIR="/home/du/new_models/TIGER-Lab/M-BEIR/" # <--- Change this to the MBEIR data directory you download from HF page


# Path to config dir
MODEL="uniir_clip/clip_scorefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
SIZE="large"
MODE="eval"  # <--- Change this to the mode you want to run
EXP_NAME="unirag"
CONFIG_DIR="{MODEL_DIR}/configs_scripts/{SIZE}/{MODE}/{EXP_NAME}"



print("PYTHONPATH: $PYTHONPATH")
print("CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES")


CONFIG_PATH="$CONFIG_DIR/embed.yaml"
SCRIPT_NAME="mbeir_embedder.py"
print("CONFIG_PATH: $CONFIG_PATH")
print("SCRIPT_NAME: $SCRIPT_NAME")

config_updater.py(update_mbeir_yaml_instruct_status, CONFIG_PATH, enable_instruct True

python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
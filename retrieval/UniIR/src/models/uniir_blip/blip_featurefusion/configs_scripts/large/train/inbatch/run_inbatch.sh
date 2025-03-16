# Train BLIPFeatureFusion model on MBEIR dataset

# Initialize Conda
source /home/miniconda3/etc/profile.d/conda.sh # <--- Change this to the path of your conda.sh

# Path to the codebase and config file
SRC="$HOME/UniIR/src"  # Absolute path to codebse /UniIR/src # <--- Change this to the path of your UniIR/src

# Path to common dir
COMMON_DIR="$SRC/common"

# Path to MBEIR data and UniIR directory where we store the checkpoints, embeddings, etc.
UNIIR_DIR="/data/UniIR/" # <--- Change this to the MBEIR directory
MBEIR_DATA_DIR="/data/UniIR/M-BEIR/" # <--- Change this to the MBEIR data directory you download from HF page

# Path to config dir
MODEL="uniir_blip/blip_featurefusion"  # <--- Change this to the model you want to run
MODEL_DIR="$SRC/models/$MODEL"
UNION_MODEL_DIR="$SRC/models/uniir_blip"
SIZE="large"
MODE="train"  # <--- Change this to the mode you want to run
EXP_NAME="inbatch"
CONFIG_DIR="$MODEL_DIR/configs_scripts/$SIZE/$MODE/$EXP_NAME"

# Set CUDA devices and PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # <--- Change this to the CUDA devices you want to us
NPROC=8
export PYTHONPATH=$SRC
echo "PYTHONPATH: $PYTHONPATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Update config
CONFIG_PATH="$CONFIG_DIR/inbatch.yaml"
cd $COMMON_DIR
python config_updater.py \
    --update_mbeir_yaml_instruct_status \
    --mbeir_yaml_file_path $CONFIG_PATH \
    --enable_instruct True

# Change to model directory
cd $UNION_MODEL_DIR
SCRIPT_NAME="train.py"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "SCRIPT_NAME: $SCRIPT_NAME"

# Activate conda environment
# conda activate blip
conda activate uniir # <--- Change this to the name of your conda environment

# Run training command
python -m torch.distributed.run --nproc_per_node=$NPROC $SCRIPT_NAME \
    --config_path "$CONFIG_PATH" \
    --uniir_dir "$UNIIR_DIR" \
    --mbeir_data_dir "$MBEIR_DATA_DIR"
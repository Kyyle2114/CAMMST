#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Exit on any error
set -e

# Function to print colored output
print_info() {
    echo -e "\033[1;34m[INFO]\033[0m $1"
}

# --- Configuration ---
DATASET_NAME="her2st"
CONFIG_FILE="./config/default.yaml"
DATASET_PATH="./data/${DATASET_NAME}"
LOG_FILE="./logs/train_${DATASET_NAME}_$(date +%Y%m%d_%H%M%S).log"
# --- End of Configuration ---

print_info "Starting CAMMST training..."
print_info "Config: ${CONFIG_FILE}"
print_info "Dataset: ${DATASET_PATH}"
print_info "Log file: ${LOG_FILE}"

mkdir -p ./logs

python main.py \
    --config "${CONFIG_FILE}" \
    --set data.dataset_path="${DATASET_PATH}" \
    --set general.output_dir="./output_dir/${DATASET_NAME}" \
    >> "${LOG_FILE}" 2>&1

# Slide inference for each visible ratio
for VISIBLE_RATIO in 0.0 0.1 0.3; do
    python inference.py \
        --config "./output_dir/${DATASET_NAME}/fold_0/config.yaml" \
        --output_dir "./output_dir/${DATASET_NAME}" \
        --visible_ratio "${VISIBLE_RATIO}" \
        --inference_output_dir "./inference_results/${DATASET_NAME}/${VISIBLE_RATIO}" \
        >> "${LOG_FILE}" 2>&1
done

print_info "Training completed! Check ${LOG_FILE} for details."
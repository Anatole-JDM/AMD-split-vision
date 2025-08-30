#!/bin/bash

#SBATCH -J LLaVABench_full_3x3
#SBATCH -o /work1/aroger/%u/job_logs/%x-%j.out
#SBATCH -e /work1/aroger/%u/job_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p mi2508x
#SBATCH -N 1


module load rocm/5.4.2
module load openmpi4/4.1.5
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate robin

MODEL_NAME="full_split_3x3_ViT_SO400M_14_SigLIP_384"
CHECKPOINT_PATH="/work1/aroger/anatolejdm/checkpoints/$MODEL_NAME/finetune_split"
MODEL_BASE="/work1/aroger/shared/downloaded_models/OpenHermes-2.5-Mistral-7B"
DEVICE="cuda:0"
CONV_MODE="vicuna_v1"
TEMPERATURE=0
MAX_NEW_TOKENS=100
IMAGE_PATH="/work1/aroger/shared/data/evals/llava-bench-in-the-wild/images"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run the Python script with the parameters
python /work1/aroger/anatolejdm/robin/robin/eval/infer_LLaVABench.py \
    --conv_mode "$CONV_MODE" \
    --temperature "$TEMPERATURE" \
    --model_base "$MODEL_BASE" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --images "$IMAGE_PATH" \
    --query_path "/work1/aroger/shared/data/evals/llava-bench-in-the-wild/questions.jsonl" \
    --output_file "/work1/aroger/shared/data/evals/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl" \

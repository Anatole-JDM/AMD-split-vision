#!/bin/bash

#SBATCH -J robin 
#SBATCH -o /work1/aroger/%u/job_logs/%x-%j.out
#SBATCH -e /work1/aroger/%u/job_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p mi2508x
#SBATCH -N 1
# mi2104x
# mi2508x

# only change this
NAME=split_3x3_ViT_SO400M_14_SigLIP_384
MODEL=OpenHermes-2.5-Mistral-7B
VISION=SplitVisionTransformer3x3
GAS=8

# don't change this
DOWNLOADED_MODEL_PATH=$WORK/downloaded_models
MODEL=$DOWNLOADED_MODEL_PATH/$MODEL
VISION='SplitVisionTransformer3x3'

TRAIN_PATH=$WORK/robin
CHECKPOINT_PATH=$WORK/checkpoints/$NAME
DATA_PATH=/work1/aroger/anatolejdm/data

module load rocm/5.4.2
module load openmpi4/4.1.5
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate robin

PRETRAIN="$CHECKPOINT_PATH/pretrain"
if [ ! -f "$PRETRAIN/mm_projector.bin" ]; then
    PRETRAIN=$(ls -dv $PRETRAIN/checkpoint-* | tail -1)
fi

# GPU_COUNT=$(rocm-smi --showuniqueid --csv | wc -l)
GPU_COUNT=8 # mi250 are "double gpus" and counted wierd so hardcoded
BATCH_SIZE=$(( 128 / GPU_COUNT / SLURM_NNODES / GAS ))

# fresh miopen cache before run (need 1 cache per node)
# important to generate hostfile in condition otherwise deepspeed will crash when only 1 node
if [ $SLURM_NNODES -gt 1 ]
then
    bash $WORK/hpc_fund_write_hostfile.sh
    while IFS= read -r node
    do
        mkdir -p "$WORK/miopen/$SLURM_JOBID/${node%% *}"
    done < $WORK/hostfiles/$SLURM_JOBID-hosts
else
    mkdir -p "$WORK/miopen/$SLURM_JOBID/$HOSTNAME"
fi

cd $TRAIN_PATH

# export MASTER_ADDR=$(head -n 1 $WORK/hostfiles/$SLURM_JOBID-hosts | cut -d ' ' -f 1)
# echo "MASTER_ADDR: $MASTER_ADDR"

export WANDB_DIR="$WORK/wandb_cache"
export WANDB_MODE="offline"
export OMP_NUM_THREADS=1

# RuntimeError: launcher 'pdsh' not installed.
    # --launcher pdsh \
    # --launcher openmpi \
deepspeed \
    --launcher slurm \
    --hostfile $WORK/hostfiles/$SLURM_JOBID-hosts \
    $TRAIN_PATH/robin/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path $MODEL \
    --version v1 \
    --data_path $DATA_PATH/generate_model.json\
    --split_shape 3x3 \
    --image_folder $DATA_PATH \
    --vision_tower $VISION \
    --finetune_ve True \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5\
    --pretrain_mm_mlp_adapter $PRETRAIN/mm_projector.bin \
    --group_by_modality_length True \
    --image_aspect_ratio pad \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --fp16 True \
    --output_dir $CHECKPOINT_PATH/finetune_split \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GAS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb \
    --vision_lr 5e-5 \

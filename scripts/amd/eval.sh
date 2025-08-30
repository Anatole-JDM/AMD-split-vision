#!/bin/bash

#SBATCH -J 2x2_full
#SBATCH -o /work1/aroger/%u/job_logs/%x-%j.out
#SBATCH -e /work1/aroger/%u/job_logs/%x-%j.err
#SBATCH -t 12:00:00
#SBATCH -p mi2104x
#SBATCH -N 1

conda init

module load rocm/5.4.2
module load openmpi4/4.1.5
source $WORK/miniconda3/etc/profile.d/conda.sh
conda activate robin

python -m pip install --upgrade pip
python -m pip install --quiet openai==0.28
python -m pip install --quiet gradio_client
python -m pip install --quiet "websockets==11.0.3"



# MODEL_PATH="/localdisks/rogeralexis/downloaded_models/$1"
MODEL_PATH="/work1/aroger/anatolejdm/checkpoints/full_split_2x2_ViT_SO400M_14_SigLIP_384/finetune_split"
MODEL_NAME=full_split_2x2_ViT_SO400M_14_SigLIP_384
conv="vicuna_v1"

export CUDA_VISIBLE_DEVICES=0


path="/work1/aroger/shared/data/evals"

SQA=true
GQA=true
TextVQA=true
VQAv2=true
MMVET=true
LLaVA=true
POPE=true

if [ "$SQA" = true ]; then
    echo "========================"
    echo "Scoring eval: SQA"
    echo "========================"

    python /work1/aroger/anatolejdm/robin/robin/eval/eval_science_qa.py \
        --base-dir $path/scienceqa/.ipynb_checkpoints \
        --result-file $path/scienceqa/answers/$MODEL_NAME.jsonl \
        --output-file $path/scienceqa/$MODEL_NAME.jsonl \
        --output-result $path/scienceqa/answers/$MODEL_NAME.json
fi

if [ "$GQA" = true ]; then
    echo "========================"
    echo "Scoring eval: GQA"
    echo "========================"

    SPLIT="llava_gqa_testdev_balanced"

    python /work1/aroger/anatolejdm/robin/scripts/convert_gqa_for_eval.py \
        --src $path/gqa/answers/$SPLIT/$MODEL_NAME/answers.jsonl \
        --dst $path/gqa/data/testdev_balanced_predictions.json

    cd $path/gqa
    python /work1/aroger/shared/data/evals/gqa/eval.py --tier data/testdev_balanced --questions /work1/aroger/shared/data/evals/gqa/data/testdev_balanced_questions.json| grep "Accuracy:"
    cd -
fi

if [ "$TextVQA" = true ]; then
    echo "========================"
    echo "Scoring eval: TextVQA"
    echo "========================"

    python /work1/aroger/anatolejdm/robin/robin/eval/eval_textvqa.py \
        --annotation-file $path/textvqa/TextVQA_0.5.1_val.json \
        --result-file $path/textvqa/answers/$MODEL_NAME.jsonl
fi

if [ "$VQAv2" = true ]; then
    echo "========================"
    echo "Scoring eval: VQAv2"
    echo "========================"

    SPLIT="llava_vqav2_mscoco_test-dev2015"

    python /work1/aroger/anatolejdm/robin/scripts/convert_vqav2_for_submission.py \
        --split $SPLIT --ckpt $MODEL_NAME --dir $path/vqav2

    echo "Submit file to website: https://eval.ai/web/challenges/challenge-page/830/submission"
    echo $path/vqav2/answers_upload/$SPLIT/$MODEL_NAME.json
fi

if [ "$MMVET" = true ]; then
    echo "========================"
    echo "Scoring eval: MMVET"
    echo "========================"

    python /work1/aroger/anatolejdm/robin/scripts/convert_mmvet_for_eval.py \
        --src $path/mm-vet/answers/$MODEL_NAME.jsonl \
        --dst $path/mm-vet/results/$MODEL_NAME.json

    python /work1/aroger/anatolejdm/robin/scripts/submit_mmvet.py \
        --src $path/mm-vet/results/$MODEL_NAME.zip \
        --dst $path/mm-vet/evaluated \
        --use-zip
fi

if [ "$LLaVA" = true ]; then
    echo "========================"
    echo "Scoring eval: LLaVA Bench"
    echo "========================"

    #define openai api key
    python /work1/aroger/anatolejdm/robin/robin/eval/eval_gpt_review_bench.py \
        --question $path/llava-bench-in-the-wild/questions.jsonl \
        --context $path/llava-bench-in-the-wild/context.jsonl \
        --rule /work1/aroger/shared/rule.json \
        --answer-list \
            $path/llava-bench-in-the-wild/answers_gpt4.jsonl \
            $path/llava-bench-in-the-wild/answers/$MODEL_NAME.jsonl \
        --output \
            $path/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl

    python /work1/aroger/anatolejdm/robin/robin/eval/summarize_gpt_review.py \
        -f $path/llava-bench-in-the-wild/reviews/$MODEL_NAME.jsonl
fi

if [ "$POPE" = true ]; then
    echo "========================"
    echo "Scoring eval: POPE"
    echo "========================"

    python /work1/aroger/anatolejdm/robin/robin/eval/eval_pope.py \
        --annotation-dir $path/pope/coco \
        --question-file $path/pope/llava_pope_test.jsonl \
        --result-file $path/pope/answers/$MODEL_NAME.jsonl
fi

echo "========================"
echo "DONE"
echo "========================"
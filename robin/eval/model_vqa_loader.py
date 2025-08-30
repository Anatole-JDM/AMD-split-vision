"""
import argparse
import os
import json
from tqdm import tqdm
import shortuuid
import torch

from robin.serve.robin_inference import Robin


def eval_model():
    # Model
    robin = Robin("/work1/aroger/anatolejdm/checkpoints/split_pretrain_2x2_ViT_SO400M_14_SigLIP_384/finetune_split",
                 model_base="/work1/aroger/anatolejdm/downloaded_models/OpenHermes-2.5-Mistral-7B",
                 device="cuda:0",
                 conv_mode="vicuna_v1",
                 temperature=0,
                 max_new_tokens=5
                )

    file_path = "/work1/aroger/shared/data/evals/scienceqa/llava_test_CQM-A.json"

    with open(file_path, "r") as f:
        questions = json.load(f)
    answers_file = "/work1/aroger/anatolejdm/checkpoints/split_pretrain_2x2_ViT_SO400M_14_SigLIP_384/inferences.json"

    ans_file = open(answers_file, "w")

    ans_file.write('[')

    img_folder = "/work1/aroger/shared/data/evals/scienceqa/images/test"


    for line in tqdm(questions):
        cur_prompt = line["conversations"][0]['value']
        try:
            image_file = os.path.join(img_folder, line["image"])

            outputs = robin(image_file, 'Answer using a single letter. Do not write any words. The questions are multiple choice, just select the letter that corresponds to the correct answer, A,B,C or D.' + cur_prompt)
        except:
            outputs = robin(None ,'Answer using a single letter. Do not write any words. The questions are multiple choice, just select the letter that corresponds to the correct answer, A,B,C or D.' + cur_prompt )
        line["conversations"][1]["value"] = outputs
        ans_file.write(json.dumps(line, indent=4, ensure_ascii=False) + ',\n')

        ans_file.flush()
    ans_file.close()
    
    with open(answers_file, 'r+', encoding="utf-8") as ans_file:
        content = ans_file.read()
        ans_file.seek(0)
        ans_file.write(content.rstrip(",\n"))  # Remove last comma or newline
        ans_file.truncate()


    with open(answers_file, 'a') as f1:
        f1.write('\n]')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model()

"""
import argparse
import os
import json
from robin.serve.robin_inference import Robin


def eval_model():
    # Initialize model
    robin = Robin("/work1/aroger/anatolejdm/checkpoints/split_pretrain_2x2_ViT_SO400M_14_SigLIP_384/finetune_split",
                 model_base="/work1/aroger/anatolejdm/downloaded_models/OpenHermes-2.5-Mistral-7B",
                 device="cuda:0",
                 conv_mode="vicuna_v1",
                 temperature=0,
                 max_new_tokens=5
                )

    file_path = "/work1/aroger/shared/data/evals/scienceqa/llava_test_CQM-A.json"

    # Load the questions file
    with open(file_path, "r") as f:
        questions = json.load(f)

    answers_file = "/work1/aroger/anatolejdm/checkpoints/split_pretrain_2x2_ViT_SO400M_14_SigLIP_384/inferences.json"
    
    # Open answers file for writing
    with open(answers_file, "w", encoding="utf-8") as ans_file:
        ans_file.write('[')  # Start the JSON array

        # Iterate through all questions
        for i in range(1,2):
            line = questions[i]
            cur_prompt = line["conversations"][0]['value']
            img_folder = "/work1/aroger/shared/data/evals/scienceqa/images/test"
            
            try:
                image_file = os.path.join(img_folder, line["image"])
                # Infer using the image
                outputs = robin(image_file, 'Answer using a single letter. Do not write any words. The questions are multiple choice, just select the letter that corresponds to the correct answer, A,B,C or D.' + cur_prompt)
            except:
                # If no image, infer without the image
                outputs = robin(None, 'Answer using a single letter. Do not write any words. The questions are multiple choice, just select the letter that corresponds to the correct answer, A,B,C or D.' + cur_prompt)

            # Store the output
            line["conversations"][1]["value"] = outputs

            # Write the result for this question, ensuring a comma is added after each entry except the last one
            ans_file.write(json.dumps(line, indent=4, ensure_ascii=False) + ',\n')

        # Write closing bracket to complete the JSON array
        ans_file.write(']')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default="")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model()
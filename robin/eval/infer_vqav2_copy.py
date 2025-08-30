from robin.serve.robin_inference import Robin
import argparse
import json
import os
import uuid
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run Robin inference on VQAV2-like dataset")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--model_base", type=str, required=True, help="Path to the base model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max new tokens to generate")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--query_path", type=str, required=True, help="Path to questions JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")

    args = parser.parse_args()
    process_questions(args)

def process_questions(args):
    # Initialize model
    robin = Robin(
        args.checkpoint_path,
        model_base=args.model_base,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        split_shape=[2,2]
    )

    # Hardcoded model ID based on your example
    MODEL_ID = "EVAL_split_2x2_ViT_SO400M_14_SigLIP_384"

    # Process questions
    with open(args.query_path, "r") as f_in, open(args.output_file, "w") as f_out:
        for line in tqdm(f_in, desc="Processing questions"):
            try:
                entry = json.loads(line.strip())
                image_path = os.path.join(args.images, entry["image"])
                
                # Use full prompt including OCR tokens
                full_prompt = entry["text"]
                
                # Get model response
                answer = robin(image_path, full_prompt).strip()
                
                # Build output entry with desired format
                output_entry = {
                    "question_id": entry["question_id"],
                    "prompt": full_prompt,
                    "text": answer,
                    "answer_id": str(uuid.uuid4()),
                    "model_id": MODEL_ID,
                    "metadata": {}
                }
                
                f_out.write(json.dumps(output_entry) + "\n")
                
            except Exception as e:
                print(f"Error processing {entry.get('question_id', 'unknown')}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
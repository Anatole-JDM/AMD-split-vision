from robin.serve.robin_inference import Robin
import argparse
import json
import os
import uuid
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Run Robin inference on SQA-like dataset")
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
    MODEL_ID = "EVAL_full_split_2x2_ViT_SO400M_14_SigLIP_384"

    with open(args.query_path, "r") as f_in:
        questions = json.load(f_in)
    
    # Process each question
    with open(args.output_file, "w") as f_out:
        for question in tqdm(questions, desc="Processing questions"):
            try:
                question_id = question.get("id", "unknown")
                
                # Get the human prompt from conversations
                if "conversations" in question and len(question["conversations"]) > 0:
                    human_message = next(
                        (msg for msg in question["conversations"] if msg["from"] == "human"), 
                        None
                    )
                    
                    if human_message:
                        full_prompt = human_message["value"]
                        
                        # Handle image if present
                        if "<image>" in full_prompt and "image" in question:
                            image_path = os.path.join(args.images, question["image"])
                            
                            # Get model response with image
                            answer = robin(image_path, full_prompt.replace("<image>", "")).strip()
                        else:
                            # Text-only question
                            answer = robin(None, full_prompt).strip()
                            
                        # Build output entry with desired format
                        output_entry = {
                            "question_id": question_id,
                            "prompt": full_prompt,
                            "text": answer,
                            "answer_id": str(uuid.uuid4()),
                            "model_id": MODEL_ID,
                            "metadata": {}
                        }
                        
                        # Get ground truth if available
                        if len(question["conversations"]) > 1:
                            gpt_message = next(
                                (msg for msg in question["conversations"] if msg["from"] == "gpt"),
                                None
                            )
                            if gpt_message:
                                output_entry["ground_truth"] = gpt_message["value"]
                        
                        f_out.write(json.dumps(output_entry) + "\n")
                    
            except Exception as e:
                print(f"Error processing question ID {question_id}: {str(e)}")
                continue

if __name__ == "__main__":
    main()
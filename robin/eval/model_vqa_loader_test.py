from robin.serve.robin_inference import Robin
import argparse
import json
import os

def main():
    parser = argparse.ArgumentParser(description="Run Robin inference on an image with a prompt.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--model_base", type=str, required=True, help="Path to the base model")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument("--conv_mode", type=str, default="vicuna_v1", help="Conversation mode")
    parser.add_argument("--temperature", type=float, default=0, help="Sampling temperature")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of new tokens to generate")
    parser.add_argument("--images", type=str, required=True, help="Path to the images directory")
    parser.add_argument("--query_path", type=str, required=True, help="Path to the query file")

    args = parser.parse_args()
    eval(args)

def eval(args):
    # Initialize Robin model
    robin = Robin(
        args.checkpoint_path,
        model_base=args.model_base,
        device=args.device,
        conv_mode=args.conv_mode,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        split_shape=[2, 2]
    )

    # Load JSON file
    try:
        with open(args.query_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Load entire JSON content
    except json.JSONDecodeError as e:
        print(f"Error loading JSON file: {e}")
        return

    output_data = []  # Store processed results

    # Process each entry
    for entry in data:
        image_path = os.path.join(args.images, entry["image"])  # Construct full image path
        conversations = entry.get("conversations", [])

        print(f"\nProcessing Image: {image_path}")

        new_conversations = []  # Store modified conversations

        for conv in conversations:
            new_conversations.append(conv)  # Keep original human conversation

            if conv["from"] == "human":  # Process only human prompts
                prompt = conv["value"].replace("<image>", "").replace("\n", "").strip()  # Clean prompt
                result = robin(image_path, prompt)  # Run inference

                print(f"Prompt: {prompt}")
                print(f"Result (writing to answers.json): {result}\n")

                # Append AI response under "split"
                new_conversations.append({"from": "split", "value": result})

        # Append processed entry
        output_data.append({
            "id": entry["id"],
            "image": entry["image"],  # Keep original image path format
            "conversations": new_conversations
        })

    # Save results to answers.json
    with open(args.checkpoint_path, "w", encoding="utf-8") as output_file:
        json.dump(output_data, output_file, indent=2, ensure_ascii=False)

    print(f"Results saved to {args.output_path}")

if __name__ == "__main__":
    main()

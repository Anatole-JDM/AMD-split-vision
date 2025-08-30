import json

input_file = "/work1/aroger/shared/data/evals/pope/answers/split_3x3_ViT_SO400M_14_SigLIP_384.jsonl"   # Change to your actual file path
output_file = "/work1/aroger/shared/data/evals/pope/answers/split_3x3_ViT_SO400M_14_SigLIP_384_fixed.jsonl"

with open(input_file, "r") as infile, open(output_file, "w") as outfile:
    for line in infile:
        data = json.loads(line.strip())  # Load each JSON object
        reformatted_data = {
            "question_id": data["question_id"],
            "label": data["text"]  # Rename "text" to "label"
        }
        outfile.write(json.dumps(reformatted_data) + "\n")  # Write in correct format

print("✅ Reformatted file saved as:", output_file)

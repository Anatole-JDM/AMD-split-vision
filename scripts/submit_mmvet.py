import zipfile
import os
import shutil
import argparse
import tempfile
from gradio_client import Client

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True,
                        help="Either: JSON of model responses (to send to evaluator) OR path to a .zip results file")
    parser.add_argument('--dst', type=str, required=True,
                        help="Location to save the evaluation results to")
    parser.add_argument('--use-zip', action='store_true',
                        help="If set, --src is treated as a zip file path instead of JSON to submit")
    args = parser.parse_args()

    score_path = args.dst
    os.makedirs(score_path, exist_ok=True)

    if args.use_zip:
        # Directly use the provided zip file
        result_file = args.src
    else:
        # Call the evaluator to generate the zip file
        file = args.src
        client = Client("whyu/MM-Vet_Evaluator")
        result_file = client.predict(
            file,
            api_name="/grade"
        )

    # Use a temporary directory for extraction
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(result_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Move extracted files into destination
        for f in os.listdir(tmpdir):
            filename = os.path.join(tmpdir, f)

            if f.endswith(".csv"):
                with open(filename, 'r') as fh:
                    lines = fh.readlines()
                    if len(lines) > 1:
                        print(lines[1].strip())
                destination = os.path.join(score_path, f)
                shutil.move(filename, destination)

            elif f.endswith(".json"):
                destination = os.path.join(score_path, f)
                shutil.move(filename, destination)

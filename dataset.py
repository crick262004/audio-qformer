# This script preprocesses the dataset by consolidating individual JSON metadata
# files from shard directories (train, dev, test) into three master JSON Lines (.jsonl) files.
# It also converts the relative audio paths in the metadata to absolute paths to ensure
# they can be reliably located by the training script. Finally, it includes a verification
# step to check that all specified audio files exist on the filesystem.

""""
.FLAC file, which is an audio file format that stands for Free Lossless Audio Codec

To break it down:
Are the audio files the .flac files?

Yes. As shown in the README's directory structure, the actual audio data is stored in the .flac files (e.g., train-000001.flac). FLAC is a popular lossless audio format.

Is the raw dataset a bunch of paired .json and .flac files?

Exactly. The raw dataset is organized into pairs. For every audio sample, there is one .json file containing the metadata (like the summary text) and one .flac file containing the corresponding audio. They share the same base filename (e.g., train-000001).

Will all lines in the .jsonl file point to a unique .flac file?

Yes, that's the whole point of the dataset.py script. The script reads each individual .json file, finds the name of its associated .flac file, turns that name into a full, absolute path, and then writes the entire metadata object (including the new absolute path) as a single line to the .jsonl file.

So, the process is a consolidation:

Before: Many train-XXXXXX.json files and many train-XXXXXX.flac files.

After: One train.jsonl file, where each line contains the information from one of the original .json files but now points directly to the absolute path of its unique .flac partner.
"""

import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

def create_jsonl_files(data_dir="data/shards", output_dir="."):
    """
    Finds individual JSON files in split subdirectories, converts audio file paths
    to absolute paths, and aggregates them into train.jsonl, dev.jsonl, and test.jsonl files.

    Args:
        data_dir (str): The root directory containing the data shards (e.g., 'data/shards').
        output_dir (str): The directory where the final .jsonl files will be saved.
    """
    # Use pathlib for modern, object-oriented path manipulation.
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Ensure the output directory exists; create it if it doesn't.
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define the standard dataset splits to process.
    splits = ["train", "dev", "test"]

    for split in splits:
        split_dir = data_dir / split

        # Check if the directory for the current split exists before proceeding.
        if not split_dir.exists():
            print(f"Warning: {split_dir} directory not found, skipping {split}")
            continue

        # Create a search pattern to find all JSON files for the current split.
        json_pattern = f"{split}-*.json"
        # Use glob to find all files matching the pattern.
        json_files = list(split_dir.glob(json_pattern))

        if not json_files:
            print(f"Warning: No JSON files found in {split_dir}")
            continue

        print(f"Processing {len(json_files)} files for {split} split...")

        # Define the output file path (e.g., 'train.jsonl').
        output_file = output_dir / f"{split}.jsonl"

        # Open the output .jsonl file for writing.
        with open(output_file, "w", encoding="utf-8") as outfile:
            # Wrap the loop with tqdm to display a progress bar.
            for json_file in tqdm(json_files, desc=f"Creating {split}.jsonl"):
                try:
                    # Open and load the data from an individual JSON metadata file.
                    with open(json_file, "r", encoding="utf-8") as infile:
                        data = json.load(infile)

                    # Get the relative audio filename from the JSON data.
                    audio_filename = data.get("audio", "")
                    if audio_filename:
                        # Construct the full path to the audio file.
                        audio_path = split_dir / audio_filename
                        # IMPORTANT: Replace the relative filename with its absolute path.
                        # This makes the dataset self-contained and easy to use from any working directory.
                        data["audio"] = str(audio_path.absolute())

                    # Write the modified dictionary as a single line (JSON object) to the output file.
                    # ensure_ascii=False is important for handling non-English characters correctly.
                    json.dump(data, outfile, ensure_ascii=False)
                    # Write a newline character to adhere to the JSON Lines format.
                    outfile.write("\n")

                except Exception as e:
                    # Catch potential errors during file processing and report them.
                    print(f"Error processing {json_file}: {e}")
                    continue

        print(f"Created {output_file} with {len(json_files)} entries")

    print("Dataset creation completed!")


def verify_audio_files(jsonl_file):
    """
    Performs a sanity check by reading a .jsonl file and verifying that every
    audio path listed within it points to an existing file on disk.

    Args:
        jsonl_file (str): Path to the .jsonl file to verify.
    """
    missing_files = []
    total_files = 0

    print(f"Verifying audio files in {jsonl_file}...")

    # Open the generated .jsonl file for reading.
    with open(jsonl_file, "r", encoding="utf-8") as f:
        # Iterate over each line in the file, showing a progress bar.
        for line_num, line in enumerate(tqdm(f, desc="Verifying files"), 1):
            try:
                # Parse the JSON object from the current line.
                data = json.loads(line.strip())
                audio_path = data.get("audio", "")
                total_files += 1

                # Use os.path.exists to check if the file is present at the specified absolute path.
                if not os.path.exists(audio_path):
                    missing_files.append((line_num, audio_path))

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")

    # After checking all lines, report the results.
    if missing_files:
        print(f"Warning: {len(missing_files)} missing audio files out of {total_files}")
        # To avoid flooding the console, only show the first 10 missing files.
        for line_num, path in missing_files[:10]:
            print(f"  Line {line_num}: {path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print(f"All {total_files} audio files found!")


# This block is the main entry point of the script.
if __name__ == "__main__":
    # Step 1: Call the function to create the .jsonl files from the raw shard data.
    create_jsonl_files("data/shards", ".")

    # Step 2: After creating the files, automatically verify each one.
    for split in ["train", "dev", "test"]:
        jsonl_file = f"{split}.jsonl"
        if os.path.exists(jsonl_file):
            verify_audio_files(jsonl_file)
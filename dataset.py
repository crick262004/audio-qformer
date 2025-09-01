import os
import json
import glob
from pathlib import Path
from tqdm import tqdm

def create_jsonl_files(data_dir="data/shards", output_dir="."):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev", "test"]

    for split in splits:
        split_dir = data_dir / split

        if not split_dir.exists():
            print(f"Warning: {split_dir} directory not found, skipping {split}")
            continue

        json_pattern = f"{split}-*.json"
        json_files = list(split_dir.glob(json_pattern))

        if not json_files:
            print(f"Warning: No JSON files found in {split_dir}")
            continue

        print(f"Processing {len(json_files)} files for {split} split...")

        output_file = output_dir / f"{split}.jsonl"

        with open(output_file, "w", encoding="utf-8") as outfile:
            for json_file in tqdm(json_files, desc=f"Creating {split}.jsonl"):
                try:
                    with open(json_file, "r", encoding="utf-8") as infile:
                        data = json.load(infile)

                    audio_filename = data.get("audio", "")
                    if audio_filename:
                        audio_path = split_dir / audio_filename
                        data["audio"] = str(audio_path.absolute())

                    json.dump(data, outfile, ensure_ascii=False)
                    outfile.write("\n")

                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue

        print(f"Created {output_file} with {len(json_files)} entries")

    print("Dataset creation completed!")


def verify_audio_files(jsonl_file):
    missing_files = []
    total_files = 0

    print(f"Verifying audio files in {jsonl_file}...")

    with open(jsonl_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(tqdm(f, desc="Verifying files"), 1):
            try:
                data = json.loads(line.strip())
                audio_path = data.get("audio", "")
                total_files += 1

                if not os.path.exists(audio_path):
                    missing_files.append((line_num, audio_path))

            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")

    if missing_files:
        print(f"Warning: {len(missing_files)} missing audio files out of {total_files}")
        for line_num, path in missing_files[:10]:
            print(f"  Line {line_num}: {path}")
        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more")
    else:
        print(f"All {total_files} audio files found!")


if __name__ == "__main__":
    create_jsonl_files("data/shards", ".")

    for split in ["train", "dev", "test"]:
        jsonl_file = f"{split}.jsonl"
        if os.path.exists(jsonl_file):
            verify_audio_files(jsonl_file)
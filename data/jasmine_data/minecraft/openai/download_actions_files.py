import subprocess
import json
import tyro
from dataclasses import dataclass
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



@dataclass
class Args:
    # index_file: str = "data/open_ai_index_files/all_6xx_Jun_29.json"
    # index_file: str = "data/open_ai_index_files/all_7xx_Apr_6.json"
    # index_file: str = "data/open_ai_index_files/all_8xx_Jun_29.json"
    # index_file: str = "data/open_ai_index_files/all_9xx_Jun_29.json"
    index_file: str = "data/open_ai_index_files/all_10xx_Jun_29.json"
    output_dir: str = "data/open_ai_minecraft_actions_files"
    num_workers: int = -1  # -1 means use all available cores


def flatten_path(relpath):
    """Convert nested path to flattened filename with subdirectory as prefix
    e.g. data/6.10/filename.mp4 -> 6.10_filename.mp4
    """

    parts = relpath.split("/")

    if len(parts) >= 3:
        subdir = parts[1]
        filename = parts[2]
        return f"{subdir}_{filename}"
    else:
        return relpath.replace("/", "_")


def download_file(args):
    try:
        url, base_dir, output_dir = args
        jsonl_url = url.rsplit(".", 1)[0] + ".jsonl"
        filename = flatten_path(jsonl_url)
        output_file = os.path.join(output_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return {"file": jsonl_url, "success": True, "skipped": True}
        
        subprocess.run(
            ["wget", "-q", base_dir + jsonl_url, "-O", output_file], check=True
        )
        return {"file": jsonl_url, "success": True, "skipped": False}
    except subprocess.CalledProcessError as e:
        # delete file if it exists
        if os.path.exists(output_file):
            os.remove(output_file)
        return {"file": jsonl_url, "success": False, "error": str(e)}


def download_actions_files(index_file: str, output_dir: str, num_workers: int):
    # load json file
    with open(index_file, "r") as f:
        data = json.load(f)

    base_dir = data["basedir"]
    urls = data["relpaths"]

    # Prepare arguments for each process
    args_list = [(url, base_dir, output_dir) for url in urls]

    results = []
    with tqdm(total=len(args_list), desc="Downloading actions files") as pbar:
        with Pool(processes=num_workers) as pool:
            for result in pool.imap_unordered(download_file, args_list):
                results.append(result)
                pbar.update(1)

    # save results to json
    meta_data_file_name = index_file.split("/")[-1].split(".")[0] + "_metadata.json"
    with open(os.path.join(output_dir, meta_data_file_name), "w") as f:
        json.dump(results, f)

    # Count statistics
    failed = [r for r in results if not r["success"]]
    skipped = [r for r in results if r["success"] and r.get("skipped", False)]
    downloaded = [r for r in results if r["success"] and not r.get("skipped", False)]
    
    print(f"Downloaded: {len(downloaded)}")
    print(f"Skipped (already exists): {len(skipped)}")
    print(f"Failed: {len(failed)}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.num_workers == -1:
        args.num_workers = cpu_count()

    print(f"Index file: {args.index_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Number of workers: {args.num_workers}")

    download_actions_files(args.index_file, args.output_dir, args.num_workers)

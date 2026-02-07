import json
import requests
import os
import tyro
import logging
from urllib.parse import urljoin
from dataclasses import dataclass, field
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from datetime import datetime
import time
import glob


# Suffix for temporary files during download
TEMP_SUFFIX = ".downloading"

# Setup logging
def setup_logger(log_dir: str) -> logging.Logger:
    """Setup logger to write failed downloads to a file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"download_failures_{timestamp}.log")
    
    logger = logging.getLogger("download_failures")
    logger.setLevel(logging.ERROR)
    
    # File handler for failures
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger, log_file


@dataclass
class DownloadVideos:
    # index_file_path: str = "data/open_ai_index_files/all_6xx_Jun_29.json"
    # index_file_path: str = "data/open_ai_index_files/all_7xx_Apr_6.json"
    # index_file_path: str = "data/open_ai_index_files/all_8xx_Jun_29.json"
    # index_file_path: str = "data/open_ai_index_files/all_9xx_Jun_29.json"
    index_file_path: str = "data/open_ai_index_files/all_10xx_Jun_29.json"
    num_workers: int = -1  # -1 means use all available cores
    output_dir: str = "data/minecraft_videos_10/"
    log_dir: str = "data/download_logs/"  # Directory for failure logs
    cleanup_incomplete: bool = True  # Clean up incomplete files on startup
    dry_run: bool = False  # Only calculate total size, don't download


def cleanup_incomplete_downloads(output_dir: str) -> int:
    """
    Remove incomplete download files (files with .downloading suffix).
    Returns the number of files cleaned up.
    """
    pattern = os.path.join(output_dir, f"*{TEMP_SUFFIX}")
    incomplete_files = glob.glob(pattern)
    
    for filepath in incomplete_files:
        try:
            os.remove(filepath)
        except OSError:
            pass
    
    return len(incomplete_files)


def get_file_size(args) -> tuple[str, int, bool, str]:
    """
    Get file size using HEAD request without downloading.
    Returns (relpath, size_in_bytes, already_exists, error_message).
    """
    relpath, url, output_path = args
    
    # Check if already downloaded
    if os.path.exists(output_path):
        local_size = os.path.getsize(output_path)
        return (relpath, local_size, True, None)
    
    try:
        response = requests.head(url, timeout=10, allow_redirects=True)
        if response.status_code == 200:
            size = response.headers.get('Content-Length')
            return (relpath, int(size) if size else 0, False, None)
        else:
            return (relpath, 0, False, f"HTTP {response.status_code}")
    except requests.exceptions.Timeout:
        return (relpath, 0, False, "Connection timeout")
    except requests.exceptions.ConnectionError as e:
        return (relpath, 0, False, f"Connection error: {str(e)[:50]}")
    except Exception as e:
        return (relpath, 0, False, f"Error: {str(e)[:50]}")


def calculate_total_size(mp4_files: list, num_workers: int, output_dir: str, log_dir: str = "data/download_logs/"):
    """
    Calculate total download size using HEAD requests.
    Shows breakdown of already downloaded vs pending.
    Logs failed checks to file.
    """
    # Setup logger for dry run failures
    logger, log_file = setup_logger(log_dir)
    
    print(f"\nCalculating total size for {len(mp4_files)} files...")
    print(f"Using {num_workers} workers for size checks")
    print(f"Failure log: {log_file}")
    
    total_size = 0
    already_downloaded_size = 0
    pending_size = 0
    already_downloaded_count = 0
    pending_count = 0
    failed_count = 0
    failed_files = []
    
    with tqdm(total=len(mp4_files), desc="Checking sizes", unit="files") as pbar:
        with Pool(processes=num_workers) as pool:
            for relpath, size, exists, error in pool.imap_unordered(get_file_size, mp4_files):
                total_size += size
                if exists:
                    already_downloaded_size += size
                    already_downloaded_count += 1
                elif size > 0:
                    pending_size += size
                    pending_count += 1
                else:
                    failed_count += 1
                    error_msg = f"Failed to check {relpath}: {error}"
                    failed_files.append(error_msg)
                    logger.error(error_msg)
                pbar.update(1)
    
    # Convert to human-readable format
    def format_size(size_bytes):
        if size_bytes >= 1024**4:
            return f"{size_bytes / (1024**4):.2f} TB"
        elif size_bytes >= 1024**3:
            return f"{size_bytes / (1024**3):.2f} GB"
        elif size_bytes >= 1024**2:
            return f"{size_bytes / (1024**2):.2f} MB"
        elif size_bytes >= 1024:
            return f"{size_bytes / 1024:.2f} KB"
        else:
            return f"{size_bytes} bytes"
    
    print(f"\n{'='*60}")
    print("SIZE SUMMARY")
    print(f"{'='*60}")
    print(f"Total files:              {len(mp4_files)}")
    print(f"Already downloaded:       {already_downloaded_count} files ({format_size(already_downloaded_size)})")
    print(f"Pending download:         {pending_count} files ({format_size(pending_size)})")
    if failed_count > 0:
        print(f"Failed to check:          {failed_count} files")
    print(f"{'='*60}")
    print(f"TOTAL SIZE:               {format_size(total_size)}")
    print(f"REMAINING TO DOWNLOAD:    {format_size(pending_size)}")
    print(f"{'='*60}")
    
    if failed_count > 0:
        print(f"\nFailed checks logged to: {log_file}")
        print(f"First 5 failures:")
        for fail in failed_files[:5]:
            print(f"  - {fail}")
    
    return total_size, pending_size


def download_single_file(args):
    """
    Download a single file with atomic write support.
    
    Downloads to a temporary file first, then renames on success.
    This ensures incomplete downloads don't leave corrupt files.
    """
    relpath, url, output_path = args
    temp_path = output_path + TEMP_SUFFIX

    # Skip if already successfully downloaded
    if os.path.exists(output_path):
        return f"Skipped {relpath} (already exists)"

    # Clean up any existing temp file from previous attempt
    if os.path.exists(temp_path):
        try:
            os.remove(temp_path)
        except OSError:
            pass

    try:
        response = requests.get(url, stream=True, timeout=60)
        if response.status_code == 200:
            # Get expected file size if available
            expected_size = response.headers.get('Content-Length')
            expected_size = int(expected_size) if expected_size else None
            
            file_size = 0
            # Download to temporary file
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        file_size += len(chunk)

            # Verify file size if Content-Length was provided
            if expected_size is not None and file_size != expected_size:
                os.remove(temp_path)
                return f"Failed {relpath}: Size mismatch (got {file_size}, expected {expected_size})"
            
            # Atomic rename: only happens if download was complete
            os.rename(temp_path, output_path)

            # Convert to MB for logging
            file_size_mb = file_size / (1024 * 1024)
            return f"Downloaded {relpath} ({file_size_mb:.2f} MB)"
        else:
            return f"Failed to download {relpath}: HTTP {response.status_code}"
    except requests.exceptions.RequestException as e:
        # Clean up partial download on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return f"Request failed for {relpath}: {e}"
    except Exception as e:
        # Clean up partial download on error
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return f"Unexpected error downloading {relpath}: {e}"


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


def download_dataset(index_file_path, output_dir, num_workers=64, dry_run=False, log_dir="data/download_logs/"):
    # Load the index file
    with open(index_file_path, "r") as f:
        index_data = json.load(f)

    basedir = index_data["basedir"]
    relpaths = index_data["relpaths"]

    # Filter for mp4 files only and flatten the path structure
    mp4_files = []
    for relpath in relpaths:
        if relpath.endswith(".mp4"):
            url = urljoin(basedir, relpath)
            flattened_filename = flatten_path(relpath)
            output_path = os.path.join(output_dir, flattened_filename)
            mp4_files.append((relpath, url, output_path))

    print(f"Found {len(mp4_files)} MP4 files")

    # Dry run: only calculate sizes
    if dry_run:
        calculate_total_size(mp4_files, num_workers, output_dir, log_dir)
        return

    # Setup failure logger
    logger, log_file = setup_logger(log_dir)
    print(f"Failure log: {log_file}")
    print(f"Using {num_workers} workers for parallel downloads")

    start_time = time.time()

    if num_workers > len(mp4_files):
        num_workers = len(mp4_files)

    with tqdm(
        total=len(mp4_files), desc="Overall Download Progress", unit="files"
    ) as pbar:
        with Pool(processes=num_workers) as pool:
            results = []
            for result in pool.imap_unordered(
                download_single_file,
                [
                    (relpath, url, output_path)
                    for relpath, url, output_path in mp4_files
                ],
            ):
                results.append(result)
                pbar.update(1)
    
    # Log failures and collect failed files
    failed_results = []
    for result in results:
        if "Downloaded" not in result and "Skipped" not in result:
            failed_results.append(result)
            logger.error(result)
    
    # Print final results summary
    successful_downloads = sum(1 for r in results if "Downloaded" in r)
    skipped_files = sum(1 for r in results if "Skipped" in r)
    failed_downloads = len(failed_results)

    print(f"\nDownload Summary:")
    print(f"  Successful downloads: {successful_downloads}")
    print(f"  Skipped files: {skipped_files}")
    print(f"  Failed downloads: {failed_downloads}")
    
    if failed_downloads > 0:
        print(f"\n  Failed downloads logged to: {log_file}")
        print(f"  First 5 failures:")
        for fail in failed_results[:5]:
            print(f"    - {fail}")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\nDownload completed in {total_time:.2f} seconds")


if __name__ == "__main__":
    args = tyro.cli(DownloadVideos)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_workers == -1:
        args.num_workers = cpu_count()

    print(f"Index file path: {args.index_file_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Log directory: {args.log_dir}")
    print(f"Number of workers: {args.num_workers}")

    if args.dry_run:
        print(f"Mode: DRY RUN (size check only)")
    else:
        # Clean up incomplete downloads from previous runs
        if args.cleanup_incomplete:
            cleaned = cleanup_incomplete_downloads(args.output_dir)
            if cleaned > 0:
                print(f"Cleaned up {cleaned} incomplete download(s) from previous run")

    download_dataset(args.index_file_path, args.output_dir, args.num_workers, args.dry_run, args.log_dir)

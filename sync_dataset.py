import os
from pathlib import Path
import sys

# --- Configuration ---
# Make sure this matches the directory you are training on
BASE_DIR = Path("data/celeb_div2k_imgnet")
SPLITS = ["train", "val", "test"]
# Update this if you are using other scales, e.g., ["X2", "X4", "X8"]
SCALES = ["X4"] 
# ---------------------

print(f"Starting dataset synchronization for: {BASE_DIR}")
if not BASE_DIR.is_dir():
    print(f"Error: Base directory not found at '{BASE_DIR}'.")
    print("Please make sure this path is correct and you have run the merge script.")
    sys.exit(1)

total_hr_removed = 0
total_lr_removed = 0

for split in SPLITS:
    for scale in SCALES:
        hr_dir = BASE_DIR / split / scale / "HR"
        lr_dir = BASE_DIR / split / scale / "LR"

        if not hr_dir.is_dir() or not lr_dir.is_dir():
            print(f"\nSkipping {split}/{scale}: One or both directories not found.")
            continue

        print(f"\n--- Processing {split}/{scale} ---")

        try:
            # Get all filenames from both directories
            hr_files = set(f.name for f in hr_dir.iterdir() if f.is_file())
            lr_files = set(f.name for f in lr_dir.iterdir() if f.is_file())

            print(f"Found {len(hr_files)} files in HR and {len(lr_files)} files in LR.")

            # Find files that are in HR but not in LR
            orphans_hr = hr_files - lr_files
            # Find files that are in LR but not in HR
            orphans_lr = lr_files - hr_files

            if not orphans_hr and not orphans_lr:
                print("Directories are already in sync. No files removed.")
                continue

            # Remove orphan HR files
            for fname in orphans_hr:
                (hr_dir / fname).unlink()
                total_hr_removed += 1
            if orphans_hr:
                print(f"Removed {len(orphans_hr)} orphan HR files (e.g., {list(orphans_hr)[0]}).")


            # Remove orphan LR files (this is the one causing your error)
            for fname in orphans_lr:
                (lr_dir / fname).unlink()
                total_lr_removed += 1
            if orphans_lr:
                print(f"Removed {len(orphans_lr)} orphan LR files (e.g., {list(orphans_lr)[0]}).")

            print(f"Sync complete for {split}/{scale}.")

        except Exception as e:
            print(f"Error processing {split}/{scale}: {e}")

print("\n--- Synchronization Finished ---")
print(f"Total orphan HR files removed: {total_hr_removed}")
print(f"Total orphan LR files removed: {total_lr_removed}")
print("Your dataset is now clean. Please re-run your training script.")
from TabDataReprGen import main
from multiprocessing import Pool
import sys

# number of files to process overall
num_filenames = 360
modes = ["c","m","cm","s"]


if __name__ == "__main__":
    # Process one mode at a time to reduce disk pressure
    # Use 4 workers instead of 11 to reduce memory/disk spike
    pool = Pool(4)
    
    for mode in modes:
        print(f"\n{'='*60}")
        print(f"Processing mode: {mode}")
        print(f"{'='*60}")
        filename_indices = list(range(num_filenames))
        mode_list = [mode] * num_filenames
        results = pool.map(main, zip(filename_indices, mode_list))
        print(f"Completed mode {mode}: {len(results)} files processed\n")

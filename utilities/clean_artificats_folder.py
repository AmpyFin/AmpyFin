#!/usr/bin/env python3
import os
import shutil

def clean_artifacts(root_dir="../artifacts"):
    if not os.path.isdir(root_dir):
        raise NotADirectoryError(f"“{root_dir}” is not a directory")
    # iterate over each entry in artifacts/
    for sub in os.listdir(root_dir):
        sub_path = os.path.join(root_dir, sub)
        if os.path.isdir(sub_path):
            # remove every file or folder inside this sub‑folder
            for entry in os.listdir(sub_path):
                entry_path = os.path.join(sub_path, entry)
                if os.path.isdir(entry_path):
                    shutil.rmtree(entry_path)
                else:
                    os.remove(entry_path)
            print(f"Cleaned: {sub_path}")

if __name__ == "__main__":
    clean_artifacts()

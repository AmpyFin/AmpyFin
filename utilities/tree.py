#!/usr/bin/env python3
"""
tree.py

Prints the folder and file structure of a given directory,
ignoring any entries (files or folders) that start with a dot.
"""

import os
import sys
import argparse

def print_tree(root_path: str, prefix: str = "") -> None:
    """
    Recursively prints a tree of files and directories under root_path,
    skipping any entry whose name starts with '.'.
    
    :param root_path: Path to the directory to print.
    :param prefix: String prefix for the current recursion level (for indenting).
    """
    try:
        entries = sorted(e for e in os.listdir(root_path) if not e.startswith('.'))
    except PermissionError:
        # Skip directories for which we don't have permissions
        return

    for index, entry in enumerate(entries):
        path = os.path.join(root_path, entry)
        connector = "└── " if index == len(entries) - 1 else "├── "
        print(prefix + connector + entry)

        if os.path.isdir(path):
            extension = "    " if index == len(entries) - 1 else "│   "
            print_tree(path, prefix + extension)


def main():
    parser = argparse.ArgumentParser(
        description="Print a tree of a directory, ignoring hidden files/folders."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Root directory to start the tree (default: current directory)",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.path)
    if not os.path.isdir(root):
        print(f"Error: {root!r} is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(root)
    print_tree(root)


if __name__ == "__main__":
    main()

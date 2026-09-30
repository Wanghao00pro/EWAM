#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fix duplicated directory levels left by archive extraction.

Turns:
  /path/adjust_bottle/aloha-agilex_clean_50/aloha-agilex_clean_50/...
into:
  /path/adjust_bottle/aloha-agilex_clean_50/...
"""

import os
import shutil
from pathlib import Path


def fix_duplicate_paths(root_dir: str, dry_run: bool = False) -> None:
    """
    Fix duplicated directory levels.

    If /path/X/X/ exists, its contents are moved up into /path/X/.

    Args:
        root_dir: Dataset root directory
        dry_run: Only report, do not modify anything
    """
    root_path = Path(root_dir)
    if not root_path.exists():
        print(f"Error: path does not exist: {root_dir}")
        return

    # Walk all directories
    fixed_count = 0
    skipped_count = 0

    for parent in sorted(root_path.rglob("*")):
        if not parent.is_dir():
            continue

        # Look for a child directory with the same name as its parent
        for child in parent.iterdir():
            if child.is_dir() and child.name == parent.name:
                duplicate_dir = child

                print(f"\nFound duplicated path: {duplicate_dir}")
                print(f"  Moving contents to: {parent}")

                if dry_run:
                    print("  (dry run, skipped)")
                    fixed_count += 1
                    continue

                # Move everything in the duplicated directory up to the parent
                moved = []
                for item in duplicate_dir.iterdir():
                    dest = parent / item.name
                    if dest.exists():
                        print(f"  Skipping existing: {item.name} -> {dest}")
                        removed = False
                    else:
                        if dry_run:
                            moved.append(item.name)
                        else:
                            shutil.move(str(item), str(dest))
                            moved.append(item.name)
                            removed = True

                # Remove the now-empty duplicated directory
                if not dry_run and duplicate_dir.exists():
                    try:
                        shutil.rmtree(duplicate_dir)
                        print(f"  Removed empty directory: {duplicate_dir}")
                    except Exception as e:
                        print(f"  Warning: could not remove {duplicate_dir}: {e}")

                if moved:
                    print(f"  Moved {len(moved)} item(s): {', '.join(moved)}")
                    fixed_count += 1
                else:
                    skipped_count += 1

                # Handle only one duplicated child per parent
                break

    print("\n" + "=" * 50)
    print("Done!")
    print(f"  Fixed:   {fixed_count}")
    print(f"  Skipped: {skipped_count}")
    print("=" * 50)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Fix duplicated directory levels after archive extraction"
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Dataset root directory"
    )
    parser.add_argument(
        "--dry_run",
        "-n",
        action="store_true",
        help="Only print the paths that would be fixed, without modifying anything"
    )

    args = parser.parse_args()

    fix_duplicate_paths(args.root_dir, args.dry_run)


if __name__ == "__main__":
    main()

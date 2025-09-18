# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import pathlib
import shutil
from typing import List

from example_policies.data_ops.merger import constants as c
from example_policies.data_ops.merger.meta_manager import MetaManager


def delete_blacklisted_episodes(dataset_path: pathlib.Path, output_path: pathlib.Path, dry_run: bool = False):
    """Create a copy of the dataset with blacklisted episodes removed.

    Args:
        dataset_path: Path to the source LerobotDataset directory
        output_path: Path where the cleaned dataset copy will be created
        dry_run: If True, only show what would be deleted without actually creating the copy
    """
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
    
    if output_path.exists():
        raise FileExistsError(f"Output directory already exists: {output_path}")

    meta_manager = MetaManager()
    meta_manager.load_from_files(dataset_path)

    if not meta_manager.blacklist:
        print("No blacklisted episodes found. Copying entire dataset.")
        if not dry_run:
            shutil.copytree(dataset_path, output_path)
            print(f"Dataset copied to {output_path}")
        return

    print(
        f"Found {len(meta_manager.blacklist)} blacklisted episodes: {meta_manager.blacklist}"
    )

    if not dry_run:
        print(f"Copying dataset from {dataset_path} to {output_path}...")
        shutil.copytree(dataset_path, output_path)
        print("Dataset copied successfully.")

    # Work on the copied dataset
    working_path = output_path if not dry_run else dataset_path
    episode_dir = working_path / c.EPISODE_DIR
    video_dir = working_path / c.VIDEO_DIR
    meta_dir = working_path / c.META_DIR

    deleted_episodes = []

    for episode_idx in meta_manager.blacklist:
        episode_file = episode_dir / f"episode_{episode_idx:06d}.parquet"
        video_episode_dir = video_dir / f"episode_{episode_idx:06d}"

        files_to_delete = []

        if episode_file.exists():
            files_to_delete.append(episode_file)

        if video_episode_dir.exists():
            files_to_delete.append(video_episode_dir)

        if files_to_delete:
            if dry_run:
                print(f"Would delete episode {episode_idx}:")
                for file_path in files_to_delete:
                    print(f"  - {file_path}")
            else:
                print(f"Deleting episode {episode_idx}...")
                for file_path in files_to_delete:
                    if file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"  - Deleted directory: {file_path}")
                    else:
                        file_path.unlink()
                        print(f"  - Deleted file: {file_path}")

                deleted_episodes.append(episode_idx)
        else:
            print(f"Episode {episode_idx} files not found (already deleted?)")

    if not dry_run and deleted_episodes:
        _update_metadata_after_deletion(meta_manager, deleted_episodes, working_path)
        print(
            f"Successfully created cleaned dataset with {len(deleted_episodes)} episodes removed."
        )

    if dry_run:
        existing_episodes = len([ep for ep in meta_manager.blacklist if (dataset_path / c.EPISODE_DIR / f'episode_{ep:06d}.parquet').exists()])
        print(
            f"\nDry run complete. Would create a copy at {output_path} with {existing_episodes} episodes removed."
        )


def _update_metadata_after_deletion(
    meta_manager: MetaManager, deleted_episodes: List[int], dataset_path: pathlib.Path
):
    """Update metadata files after deleting episodes."""

    for episode_idx in deleted_episodes:
        if str(episode_idx) in meta_manager.episode_mapping:
            del meta_manager.episode_mapping[str(episode_idx)]

    meta_manager.episodes = [
        ep
        for ep in meta_manager.episodes
        if ep["episode_index"] not in deleted_episodes
    ]
    meta_manager.stats = [
        stat
        for stat in meta_manager.stats
        if stat["episode_index"] not in deleted_episodes
    ]

    if meta_manager.info:
        meta_manager.info["total_episodes"] = len(meta_manager.episodes)
        meta_manager.info["total_frames"] = sum(
            stat.get("num_frames", 0) for stat in meta_manager.stats
        )
        if meta_manager.info["total_episodes"] > 0:
            meta_manager.info["splits"]["train"] = (
                f"0:{meta_manager.info['total_episodes']}"
            )

    meta_manager.blacklist = []

    meta_manager.save(dataset_path)


def main():
    parser = argparse.ArgumentParser(
        description="Create a copy of a LerobotDataset with blacklisted episodes removed"
    )
    parser.add_argument(
        "dataset_path", type=pathlib.Path, help="Path to the source LerobotDataset directory"
    )
    parser.add_argument(
        "output_path", type=pathlib.Path, help="Path where the cleaned dataset copy will be created"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually creating the copy",
    )

    args = parser.parse_args()

    delete_blacklisted_episodes(args.dataset_path, args.output_path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()


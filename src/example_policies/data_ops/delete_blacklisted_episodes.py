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
        _renumber_episodes_and_update_metadata(meta_manager, deleted_episodes, working_path)
        print(
            f"Successfully created cleaned dataset with {len(deleted_episodes)} episodes removed."
        )

    if dry_run:
        existing_episodes = len([ep for ep in meta_manager.blacklist if (dataset_path / c.EPISODE_DIR / f'episode_{ep:06d}.parquet').exists()])
        print(
            f"\nDry run complete. Would create a copy at {output_path} with {existing_episodes} episodes removed."
        )


def _renumber_episodes_and_update_metadata(
    meta_manager: MetaManager, deleted_episodes: List[int], dataset_path: pathlib.Path
):
    """Renumber remaining episodes sequentially and update metadata."""
    episode_dir = dataset_path / c.EPISODE_DIR
    video_dir = dataset_path / c.VIDEO_DIR
    
    # Get all remaining episode files sorted by episode number
    remaining_episodes = []
    for episode in meta_manager.episodes:
        if episode["episode_index"] not in deleted_episodes:
            remaining_episodes.append(episode)
    
    remaining_episodes.sort(key=lambda x: x["episode_index"])
    
    # Create mapping from old episode index to new sequential index
    old_to_new_mapping = {}
    for new_idx, episode in enumerate(remaining_episodes):
        old_idx = episode["episode_index"]
        old_to_new_mapping[old_idx] = new_idx
    
    # Renumber episode files
    print("Renumbering episode files...")
    for old_idx, new_idx in old_to_new_mapping.items():
        if old_idx != new_idx:
            # Rename episode parquet file
            old_episode_file = episode_dir / f"episode_{old_idx:06d}.parquet"
            new_episode_file = episode_dir / f"episode_{new_idx:06d}.parquet"
            if old_episode_file.exists():
                old_episode_file.rename(new_episode_file)
                print(f"  Renamed {old_episode_file.name} -> {new_episode_file.name}")
            
            # Rename episode video directory
            old_video_dir = video_dir / f"episode_{old_idx:06d}"
            new_video_dir = video_dir / f"episode_{new_idx:06d}"
            if old_video_dir.exists():
                old_video_dir.rename(new_video_dir)
                print(f"  Renamed {old_video_dir.name} -> {new_video_dir.name}")
    
    # Update metadata with new sequential indices
    new_episode_mapping = {}
    for episode in remaining_episodes:
        old_idx = episode["episode_index"]
        new_idx = old_to_new_mapping[old_idx]
        episode["episode_index"] = new_idx
        
        # Update episode mapping if it exists
        old_key = str(old_idx)
        if old_key in meta_manager.episode_mapping:
            new_episode_mapping[str(new_idx)] = meta_manager.episode_mapping[old_key]
    
    # Update stats with new indices
    for stat in meta_manager.stats:
        if stat["episode_index"] not in deleted_episodes:
            old_idx = stat["episode_index"]
            stat["episode_index"] = old_to_new_mapping[old_idx]
    
    # Filter out deleted episodes from stats
    meta_manager.stats = [
        stat for stat in meta_manager.stats 
        if stat["episode_index"] < len(remaining_episodes)
    ]
    
    meta_manager.episodes = remaining_episodes
    meta_manager.episode_mapping = new_episode_mapping

    if meta_manager.info:
        meta_manager.info["total_episodes"] = len(remaining_episodes)
        meta_manager.info["total_frames"] = sum(
            episode["length"] for episode in remaining_episodes
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


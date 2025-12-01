#!/usr/bin/env python3
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

"""
Script to visualize policy evaluation on a dataset episode.
Compares predicted actions vs target actions and saves a plot.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Visualize policy evaluation on a dataset episode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python visualize_policy_eval.py \\
    --checkpoint outputs/train/2025-09-16/16-37-30_model/checkpoints/last/pretrained_model \\
    --dataset /data/pick_and_place_block_di16 \\
    --episode 0 \\
    --output ./viz/episode0_comparison.png
        """,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset directory",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to compare (default: 0)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for the plot (default: ./actions_episode<N>.png)",
    )

    args = parser.parse_args()

    # Get the repository root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    validate_script = repo_root / "src" / "example_policies" / "validate_with_plot.py"

    # Check if the validation script exists
    if not validate_script.exists():
        print(f"Error: Could not find validation script at {validate_script}")
        sys.exit(1)

    # Build the command
    cmd = [
        sys.executable,
        str(validate_script),
        "--checkpoint",
        str(args.checkpoint),
        "--dataset",
        str(args.dataset),
        "--episode",
        str(args.episode),
    ]

    if args.output:
        cmd.extend(["--output", str(args.output)])

    # Display info
    print("Running visualization...")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Episode: {args.episode}")
    if args.output:
        print(f"Output: {args.output}")
    else:
        print(f"Output: ./actions_episode{args.episode}.png")
    print()

    # Execute the command
    try:
        result = subprocess.run(cmd, check=True)
        sys.exit(result.returncode)
    except subprocess.CalledProcessError as e:
        print(f"Error: Visualization failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        print("\nVisualization interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

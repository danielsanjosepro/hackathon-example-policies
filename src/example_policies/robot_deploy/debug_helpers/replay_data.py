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
import time
from pathlib import Path

import grpc

# Lerobot Environment Bug
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.robot_deploy.action_translator import ActionTranslator
from example_policies.robot_deploy.policy_loader import load_metadata
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.robot_io.robot_service import robot_service_pb2_grpc
from example_policies.robot_deploy.utils import print_info
from example_policies.robot_deploy.utils.action_mode import ActionMode


class FakeConfig:
    def __init__(self, m) -> None:
        self.metadata = m
        self.output_features = {}
        self.input_features = {}
        self.input_features["observation.state"] = np.asarray(
            m["features"]["observation.state"]["names"]
        )
        self.output_features["action"] = np.asarray(m["features"]["action"]["names"])

    def get_tcp_from_state(self, state: np.ndarray) -> np.ndarray:
        state_names = []
        state_names.extend([f"tcp_left_pos_{i}" for i in "xyz"])
        state_names.extend([f"tcp_left_quat_{i}" for i in "xyzw"])
        state_names.extend([f"tcp_right_pos_{i}" for i in "xyz"])
        state_names.extend([f"tcp_right_quat_{i}" for i in "xyzw"])

        state_indices = [
            np.where(self.input_features["observation.state"] == name)[0][0]
            for name in state_names
        ]
        return state[:, state_indices]


def inference_loop(
    data_dir: Path,
    service_stub: robot_service_pb2_grpc.RobotServiceStub,
    ep_index: int = 0,
    replay_frequency: float = 5.0,
    ask_for_input: bool = True,
):
    """Replay data from a given directory on the robot.

    Args:
        data_dir (Path): Path to the data directory.
        service_stub (robot_service_pb2_grpc.RobotServiceStub): gRPC service stub.
        ep_index (int): Episode index to run.
        replay_frequency (float): Frequency to replay the data.
        ask_for_input (bool): Whether to ask for user input at each action.
    """
    fake_repo_id = data_dir.name

    meta_data = load_metadata(data_dir)

    cfg = FakeConfig(meta_data)
    dbg_printer = print_info.InfoPrinter(cfg)

    dataset = LeRobotDataset(
        repo_id=fake_repo_id,
        root=data_dir,
        episodes=[ep_index],
    )

    robot_interface = RobotInterface(service_stub, cfg)
    model_to_action_trans = ActionTranslator(cfg)

    print(f"Replaying episode {ep_index} from {data_dir}...")
    print(f"The robot interface is {robot_interface}")
    print(f"The action translator is {model_to_action_trans}")

    step = 0
    done = False

    observation = None
    while not observation:
        observation = robot_interface.get_observation("cpu")
        time.sleep(0.1)

    input("Press Enter to move robot to start...")
    robot_interface.move_home()

    input("Press Enter to continue...")

    # Inference Loop
    print("Starting inference loop...")
    period = 1.0 / replay_frequency

    while not done:
        if dataset[step]["episode_index"] != ep_index:
            step += 1
            continue

        start_time = time.time()
        observation = robot_interface.get_observation("cpu")

        if observation:
            action = dataset[step]["action"]

            if ask_for_input:
                input("Press Enter to send next action...")

            action = model_to_action_trans.translate(action, observation)
            dbg_printer.print(step, observation, action, raw_action=False)

            robot_interface.send_action(action, model_to_action_trans.action_mode)

        elapsed_time = time.time() - start_time
        sleep_duration = period - elapsed_time

        print(f"Sleep duration: {sleep_duration} s")

        # wait for input
        # input("Press Enter to continue...")
        time.sleep(max(0.0, sleep_duration))

        step += 1


def main():
    parser = argparse.ArgumentParser(description="Robot service client")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Path to the data directory",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="Robot service server address (default: localhost:50051)",
    )

    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to run (default: 0)",
    )
    parser.add_argument(
        "--replay-frequency",
        type=float,
        default=10.0,
        help="Frequency to replay the data (default: 10.0 Hz)",
    )
    parser.add_argument(
        "--continuous-replay",
        action="store_true",
        help="Whether to continuously loop over the episode and not ask for user input at each action (default: False)",
    )

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    try:
        inference_loop(
            args.data_dir,
            stub,
            args.episode,
            replay_frequency=args.replay_frequency,
            ask_for_input=not args.continuous_replay,
        )
    except Exception as e:
        print(f"Error occurred: {e}")
        raise e
    finally:
        channel.close()
        print("Connection closed.")


if __name__ == "__main__":
    main()

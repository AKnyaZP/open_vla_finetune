import websocket_policy_server
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Union
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model
)
from experiments.robot.openvla_utils import get_processor

@dataclass
class GenerateConfig:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    center_crop: bool = False # Center crop? (if trained w/ random crop image aug)
    load_in_8bit: bool = False
    load_in_4bit: bool = False

def eval_model(checkpoint, localhost):
    ###
    cfg = GenerateConfig()
    assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"

    # [OpenVLA] Set action un-normalization key
    cfg.unnorm_key = "dsynth_atomic_tasks" #language_table_sim35k
    cfg.pretrained_checkpoint = checkpoint

    # Load model
    model = get_model(cfg)
    print("MODEL LOADED")

    # [OpenVLA] Get Hugging Face processor
    processor = None
    if cfg.model_family == "openvla":
        processor = get_processor(cfg)
    print("PROCESSOR LOADED")

    # Get expected image dimensions
    resize_size = get_image_resize_size(cfg)
    print(resize_size)
    ###

    policy = model
    policy_metadata = {}
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=localhost,
        metadata=policy_metadata,
        cfg=cfg,
        processor=processor
    )
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True) #/home/alex/projects/openvla/logs/rbm12_board1
    parser.add_argument("--localhost", type=int, default=8000)
    args = parser.parse_args()
    eval_model(checkpoint=args.checkpoint, localhost=args.localhost)
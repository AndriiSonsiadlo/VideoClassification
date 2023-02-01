from dataclasses import dataclass


@dataclass(init=True, unsafe_hash=True, frozen=True, order=True)
class Action:
    action_name: str
    action_path: str
    video_number: int
    test_split: float
    train_video_paths: list[str]
    test_video_paths: list[str]

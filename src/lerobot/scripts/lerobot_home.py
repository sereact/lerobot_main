import logging
from dataclasses import asdict, dataclass
from pprint import pformat

from lerobot.configs import parser
from lerobot.robots import Robot, RobotConfig, make_robot_from_config
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig, make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_devices
from lerobot.utils.utils import init_logging


@dataclass
class SafetyStopConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig


def do_safety_stop(device: Robot | Teleoperator):
    if hasattr(device, "teleop_safety_stop"):
        device.teleop_safety_stop()
    elif hasattr(device, "safety_stop"):
        device.safety_stop()
    else:
        raise AttributeError(
            f"Robot type {getattr(device, 'robot_type', type(device))} doesn't have safety stop method"
        )


@parser.wrap()
def safety_home(cfg: SafetyStopConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))
    robot = make_robot_from_config(cfg.robot)
    teleoperator = make_teleoperator_from_config(cfg.teleop)
    try:
        robot.connect()
        teleoperator.connect()
        do_safety_stop(robot)
        do_safety_stop(teleoperator)
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
        teleoperator.disconnect()


def main():
    register_third_party_devices()
    safety_home()


if __name__ == "__main__":
    main()

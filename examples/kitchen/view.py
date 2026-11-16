"""Preview a kitchen rollout trial with G1 held in place; drag props and doors with the mouse."""
import argparse

from egox.sim import viewer
import scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scene.add_arguments(parser)
    viewer.add_arguments(parser)
    args = parser.parse_args()
    try:
        scene_usd, robot, trial = scene.viewer_setup(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    viewer.view(args, scene_usd, robot, scene.CAMERAS, trial=trial, room_path=scene.ROOM)


if __name__ == '__main__':
    main()

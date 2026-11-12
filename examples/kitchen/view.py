"""View the kitchen and the simulated G1; props and doors move with the mouse."""
import argparse

from egox.sim import viewer
import scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    scene.add_arguments(parser)
    viewer.add_arguments(parser)
    args = parser.parse_args()
    robot = None if args.robot_at == 'none' else scene.robot_at(args.robot_at)
    viewer.view(args, scene.scene_path(args), robot, scene.CAMERAS)


if __name__ == '__main__':
    main()

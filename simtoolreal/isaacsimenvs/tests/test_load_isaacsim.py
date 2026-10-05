"""Smoke test: launch Isaac Sim headless via AppLauncher."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher


def main():
    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(["--headless"])

    app_launcher = AppLauncher(args)
    app = app_launcher.app

    print("Isaac Sim load OK")

    # Kit's app.close() hangs on shutdown, so force-exit (see distill.py pattern).
    del app
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()

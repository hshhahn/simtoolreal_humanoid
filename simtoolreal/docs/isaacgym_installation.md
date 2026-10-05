# Installation

## Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Main Environment

We use Isaac Gym for policy learning in simulation, which requires Python 3.8.

```bash
# Create a virtual environment with Python 3.8
uv venv --python 3.8

# Auto-set LD_LIBRARY_PATH on activate (needed for Isaac Gym to find libpython3.8)
echo 'export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))"):$LD_LIBRARY_PATH' >> .venv/bin/activate

source .venv/bin/activate

# Install the project and all dependencies
uv pip install -e .

# Download and extract Isaac Gym Preview 4 to a directory outside of this repo
wget https://developer.nvidia.com/isaac-gym-preview-4 -O IsaacGym_Preview_4_Package.tar.gz
tar -xzf IsaacGym_Preview_4_Package.tar.gz
cd isaacgym/python
uv pip install -e .
cd -

# Install this repo's rl_games
cd rl_games
uv pip install -e .
cd -
```

## Sim2Real Env

For real-world deployment, we need ROS1 Noetic. This can either be installed globally or using RoboStack.

### Global ROS

Follow the instructions at https://wiki.ros.org/noetic/Installation/Ubuntu to install ROS1 Noetic globally. Then source the ROS environment variables in your shell:

```bash
source /opt/ros/noetic/setup.bash
```

Now, you can use this global ROS setup with the virtual environment above.

You may need to run:

```bash
uv pip install rospkg
```

### RoboStack

Note that the RoboStack option requires a separate environment because it requires Python 3.11+. Follow the instructions at https://robostack.github.io/noetic.html to create an environment with ROS1 Noetic. Then follow the installation instructions above for the main environment (skip the venv creation and Isaac Gym installation).

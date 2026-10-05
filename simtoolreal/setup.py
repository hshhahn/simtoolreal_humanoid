from setuptools import setup, find_packages

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.23.1",
    "torch",
    "omegaconf",
    "termcolor",
    "jinja2",
    "hydra-core>=1.2",
    "rl-games>=1.6.0",
    "pyvirtualdisplay",
    "urdfpy==0.0.22",
    "pysdf==0.1.9",
    "warp-lang==0.10.1",
    "trimesh==3.23.5",
]

# Installation operation
setup(
    name="simtoolreal",
    version="0.1.0",
    author="Tyler Lum, Kushal Kedia",
    author_email="tylergwlum@gmail.com, kk837@cornell.edu",
    url="https://github.com/tylerlum/simtoolreal",
    description="Official implementation of SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation",
    keywords=["robotics", "rl", "dexterous manipulation"],
    include_package_data=True,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.6, 3.7, 3.8"],
    zip_safe=False,
)

import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Installs
setuptools.setup(
    name="nuplan-zigned",
    version="1.0.0",
    author="Zigned @ Chongqing University",
    author_email="zigned@qq.com",
    description="Author's PyTorch implementation of Reinforced Imitative Trajectory Planning for Urban Automated Driving.",
    url="https://github.com/Zigned/nuplan_zigned",
    python_requires=">=3.9,<3.10",
    packages=["nuplan_zigned"],
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
        "License :: Free for non-commercial use",
    ],
    license="apache-2.0",
    install_requires=requirements,
)

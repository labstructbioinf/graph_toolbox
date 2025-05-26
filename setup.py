from setuptools import setup
import os

if not os.path.islink("graph_toolbox") and not os.path.exists("graph_toolbox"):
    os.symlink("feature", "graph_toolbox")

setup(
    name="graph_toolbox",
    version="0.1",
    packages=["graph_toolbox"],
)



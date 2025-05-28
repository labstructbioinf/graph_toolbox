from setuptools import setup, find_packages

setup(
    name="graph_toolbox",
    version="0.1",
    packages=find_packages(),  # wykryje feature jako podpakiet
    install_requires=[
        "torch",
        "torchdata==0.9.0",
        "dgl==1.1.3",
        "biopandas",
        "pydantic",
    ],
)



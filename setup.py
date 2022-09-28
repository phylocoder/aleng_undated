""" Setup script for ALENG """
from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "decorator>=5.1.1",
    "ete3>=3.1.2",
    "graphviz>=0.20.1",
    "networkx==2.8.6",
    "numba==0.56.2",
    "numpy==1.23.3",
    "PyQt5==5.15.7",
    "PyQt5-Qt5==5.15.2",
    "PyYAML==6.0",
    "scipy==1.9.1",
    "six==1.16.0"
]

with open("README.md", encoding="utf8") as f:
    README_TEXT = f.read()

with open("LICENSE", encoding="utf8") as f:
    LICENSE_TEXT = f.read()

setup(
    name="alengu",
    python_requires=">3.10.6",
    version="0.9.4",
    description="ALE NG undated",
    long_description=README_TEXT,
    author="PyloCoder",
    author_email="phylocoder@gmail.com",
    url="https://github.com/phylocoder/aleng_undated",
    license=LICENSE_TEXT,
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(exclude=("tests", "docs", "examples")),
    entry_points={"console_scripts": ["aleng_undated=alengu.main:main"],},
)

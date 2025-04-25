from setuptools import setup, find_packages

setup(
    name="agfn",
    package_dir={"": "agfn"},
    packages=find_packages("agfn"),
    python_requires=">=3.11",
)

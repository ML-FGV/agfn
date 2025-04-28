from setuptools import setup, find_packages

# Install agfn/ as a package named agfn
setup(
    name="agfn",
    package_dir={"": "."},
    packages=find_packages("."),
    python_requires=">=3.11",
)

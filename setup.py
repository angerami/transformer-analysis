from setuptools import setup, find_packages

setup(
    name="transformer-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)

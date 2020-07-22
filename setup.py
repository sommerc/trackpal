import re
from setuptools import setup

with open("README.md", "rb") as f:
    description = f.read().decode("utf-8")

with open('careless/version.py', "r") as f:
    exec(f.read())

setup(
    name = "TrackPal",
    packages = ["trackpal"],
    version = __version__,
    description = description,
    long_description = description,
    long_description_content_type='text/markdown',
    url='https://git.ist.ac.at/csommer/trackpal',
#    entry_points = {'console_scripts': []},
    author = "Christoph Sommer",
    author_email = "christoph.sommer23@gmail.com",
    install_requires=[
            "numpy",
            "pandas>=1.0.4",
            "scikit_image",
            "scikit_learn",
            "tifffile",
            "tqdm",
            "scipy",
            "statsmodel",
            "matplotlib"
      ]
    )
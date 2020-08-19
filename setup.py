import pathlib
from setuptools import setup

# The directory containing this file
_this_dir = pathlib.Path(__file__).parent

# The text of the README file
long_description = (_this_dir / "README.md").read_text()

# Exec version file
exec((_this_dir / "trackpal" / "version.py").read_text())

setup(
    name="TrackPal",
    packages=["trackpal"],
    version=__version__,
    description="TrackPal: Tracking Python AnaLyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.ist.ac.at/csommer/trackpal",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
    ],
    #    entry_points = {'console_scripts': []},
    author="Christoph Sommer",
    author_email="christoph.sommer23@gmail.com",
    install_requires=[
        "numpy",
        "pandas>=1.0.4",
        "scikit-image",
        "scikit-learn==0.21.1",
        "tifffile",
        "tqdm",
        "scipy",
        "statsmodels",
        "matplotlib",
        "rdp",
        "pingouin",
        "seaborn",
    ],
)


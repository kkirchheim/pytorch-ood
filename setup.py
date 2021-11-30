import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="oodtk",
    version="0.0.1",
    description="Toolkit for Out-of-Distribution-Detection",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Konstantin Kirchheim",
    author_email="konstantin.kirchheim@ovgu.de",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["oodtk", "pytorch"],
    include_package_data=True,
    install_requires=[
        "scikit-learn",
        "torch",
        "torchvision",
        "pandas",
        "numpy",
        "scipy",
    ],
)

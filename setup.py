from setuptools import setup, find_packages

setup(
    name="TetriumColor",
    version="0.1.0",
    description="A Python library to work in concert with Tetrium",
    author="Jessica Lee",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "colour-science>=0.1.5",
        "matplotlib>=3.9.3",
        "numpy>=2.2.0",
        "open3d>=0.18.0",
        "packcircles>=0.14",
        "pandas>=2.2.3",
        "Pillow>=11.0.0",
        "pyserial>=3.5",
        "scipy>=1.14.1",
        "screeninfo>=0.8.1",
        "setuptools>=75.1.0",
        "aepsych>=0.6.0",
        "torch>=2.0.0",
        "tqdm>=4.67.0",
    ],  # Core dependencies
)

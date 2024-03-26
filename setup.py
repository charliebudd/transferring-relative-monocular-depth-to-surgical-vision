from setuptools import find_packages, setup

setup(
    name="trmdsv",
    author="Charlie Budd",
    author_email="charles.budd@kcl.ac.uk",
    url="https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"trmdsv": "src/trmdsv"},
    install_requires=[
        "torch>=2.2.0",
        "torchvision>=0.17.0",
        "timm>=0.9.16",
    ],
)

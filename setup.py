from setuptools import setup, find_packages

setup(
    name='trmdsv',
    author='Charlie Budd',
    author_email='charles.budd@kcl.ac.uk',
    url='https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision',
    license='MIT',
    packages=find_packages(where='src', include=['trmdsv', 'trmdsv.*']),
    package_dir={'': 'src'},
    install_requires=[
        'torch>=2.2.0',
        'torchvision>=0.17.0',
        'timm>=0.9.16',
    ],
)

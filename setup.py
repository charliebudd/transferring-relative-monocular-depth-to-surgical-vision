from setuptools import setup, find_packages

setup(
    name='trmdsv',
    author='Charlie Budd',
    author_email='charles.budd@kcl.ac.uk',
    url='https://github.com/charliebudd/transferring-relative-monocular-depth-to-surgical-vision',
    license='MIT',
    packages=find_packages("src"),
    package_dir={'trmdsv': 'src/trmdsv'},
)

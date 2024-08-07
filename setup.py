from setuptools import setup, find_packages

setup(
    name='plaNETic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'corner==2.2.1',
        'matplotlib',
        'mpltern==1.0.2',
        'pandas==1.4.1',
        'pickleshare==0.7.5',
        'pip==22.1.2',
        'python-ternary',
        'scipy==1.8.1',
        'seaborn',
        'tqdm',
        'wheel',
        'datetime',
        'h5py==3.11.0',
        'numpy==1.24.4',
        'scikit-learn',
        'tensorflow-metal==0.3.0',
        'tensorflow-macos==2.7.0',
        'protobuf<=3.21'
    ],
    author='Jo Ann Egger',
    author_email='jo-ann.egger@unibe.ch',
    description='A neural network-based Bayesian internal structure modelling framework for small exoplanets',  # A short description of your package
    url='https://github.com/joannegger/plaNETic',
)
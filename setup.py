from setuptools import setup, find_packages

setup(
    name='plaNETic',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        pylab,
        numpy,
        matplotlib,
        scipy,
        seaborn,
        pandas,
        tensorflow,
        sklearn,
        datetime,
        pickle,
        corner,
        h5py,
        ternary,
        tqdm
    ],
    author='Jo Ann Egger',
    author_email='jo-ann.egger@unibe.ch',
    description='A neural network-based Bayesian internal structure modelling framework for small exoplanets',  # A short description of your package
    url='https://github.com/joannegger/plaNETic',
)
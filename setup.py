from setuptools import setup, find_packages

setup(
    name='ml_helpers',
    version='0.1',
    packages=find_packages(),
    description='Machine Learning toolkit',
    url='https://github.com/philipp-ludersdorfer/ml_helpers',
    install_requires=[
        'matplotlib.pyplot',
        'sklearn'
    ],
)
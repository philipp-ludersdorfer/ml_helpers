from setuptools import setup, find_packages

setup(
    name='ml_helpers',
    version='0.1',
    packages=find_packages(),
    description='Client Performance Reporting',
    url='http://github.com/',
    install_requires=[
        'pandas~=0.23.0'
    ],
    setup_requires=['pytest-runner'],
    tests_require=[
        'pytest',
    ],
)
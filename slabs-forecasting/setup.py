from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
with open(here / 'requirements.txt') as reqs:
    requirements = reqs.readlines()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='enalpha_analytics',
    version='0.0.1',
    description='Analytics module for Enalpha',

    long_description=long_description,
    long_description_content_type='text/markdown',

    author='aigorithmics',

    packages=find_packages(where='.'),
    python_requires='>=3.7, <4',
    install_requires=requirements,

    extras_require={
        'test': ['pytest', 'pytest-lazy-fixture'],
    },

)

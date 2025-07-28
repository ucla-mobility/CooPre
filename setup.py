# -*- coding: utf-8 -*-
# Author: Seth Z. Zhao <sethzhao@g.ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib


from os.path import dirname, realpath
from setuptools import setup, find_packages
from opencood.version import __version__


def _read_requirements_file():
    """Return the elements in requirements.txt."""
    req_file_path = '%s/requirements.txt' % dirname(realpath(__file__))
    with open(req_file_path) as f:
        return [line.strip() for line in f]

setup(
    name='CooPre',
    version=__version__,
    packages=find_packages(),
    url='https://github.com/ucla-mobility/CooPre',
    license='MIT',
    author='Seth Z. Zhao',
    author_email='sethzhao@g.ucla.edu',
    description='CooPre codebase',
    long_description=open("README.md").read(),
    install_requires=_read_requirements_file(),
)

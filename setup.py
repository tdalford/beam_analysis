#!/usr/bin/env python3

from setuptools import setup

setup(
    name='beam_analysis',
    version='0.1',
    author='Tommy Alford',
    description='Functions for analyzing and plotting near and far-field beams',
    packages=['beam_analysis'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy',
    ],
)

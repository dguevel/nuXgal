#!/usr/bin/env python
from setuptools import setup, find_packages
from version import get_git_version

setup(
    name='KIPAC_nuXgal',
    version=get_git_version(),
    author='Arka Banarjee, Eric Charles, Ke Fang, David Guevel, Yuuki Omori',
    author_email='',
    description='A Python package for analysis of neutrino galaxy cross correlations',
    license='gpl2',
    packages=['KIPAC.nuXgal'],
    include_package_data=True,
    url="https://github.com/dguevel/nuXgal",
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GPL2 License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
    ],
    scripts=[],
    entry_points={'console_scripts': [
            'nuXgal_generateICIRFS = scripts.generateICIRFS:main',
            'nuXgal = scripts.nuXgal:main',
            'nuXgal-sensitivity = scripts.sensitivity:main',
            'nuXgal-sensitivity-plot = scripts.sensitivity_plot:main',
            ]},
    install_requires=[
        'numpy >= 1.6.1',
        'astropy >= 3.2.2',
        'matplotlib >= 3.1.1',
        'scipy >= 1.3.1',
        'healpy >= 1.12.0',
        'emcee',
        'corner',
        'tqdm',
        'csky',
        'pandas'
    ],
    extras_require=dict(
        all=[],
    ),
)

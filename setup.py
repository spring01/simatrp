
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='SimATRP',
    version='0.0.1',
    description='A controllable ATRP simulator based on solving the ATRP kinetics ODEs',
    long_description=long_description,
    url='https://github.com/spring01/simatrp',
    author='Haichen Li',
    author_email='lihc2012@gmail.com',
    license='GPLv3',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Chemistry',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
    ],
    keywords='ATRP',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'gym', 'matplotlib', 'pygame', 'h5py'],
    scripts=['bin/simatrp_interactive.py',],
)

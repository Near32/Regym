from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

test_requirements = ['pytest']

setup(
    name='Archi',
    version='0.0.1',
    description='Library of modular and highly reconfigurable architectural building blocks for general-purpose Deep Learning applications. Developed by Kevin Denamganaï at the University of York.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Near32/Archi',
    author='Kevin Denamganaï',
    author_email='denamganai.kevin@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Deep Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=[
	#'tqdm',
        'ordered-set',
        'pyyaml',
        'cloudpickle',
	'numpy',
	#'scikit-image',
        # previously: 'scikit-learn==0.23',
        # now:
        #'scikit-learn==1.0',
	#'opencv-python',
	'torch>=1.4',
	'torchvision>=0.5.0',
        #'tensorboardX',
	#'matplotlib',
	#'docopt',
	] + test_requirements,

    python_requires=">=3.6",
)

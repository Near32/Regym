from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as readme:
    long_description = readme.read()

test_requirements = ['dill', 'tqdm', 'pytest',
                     'gym-rock-paper-scissors==0.1', 'gym-kuhn-poker==0.1']

setup(
    name='regym',
    version='0.0.1',
    description='Framework to carry out (Multi-Agent) Deep Reinforcement Learning experiments.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Danielhp95/Regym',
    author='IGGI PhD Programme',
    author_email='danielhp95@gmail.com',

    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence :: Reinforcement Learning',
        'Programming Language :: Python'
    ],

    packages=find_packages(),
    zip_safe=False,

    install_requires=[
      'gym==0.25',
      'ray',
      'coloredlogs',
      'scikit-image',
      'h5py',
      'moviepy',
      'imageio',
      'matplotlib',
      'docopt',
      'pyyaml',
      'pip',
      'tensorboardx',
      'opencv-python',
      # PREVIOUSLY: 'torch',#==1.8.1',
      # THEN : 'torch==1.12.0',
      # NOW:
      'torch==1.13.1',
      'torchvision==0.14.1',
      'cvxopt',
      'scipy',
      #'minerl',
      #'iglu',
      #'gridworld',
      'celluloid',
      'scikit-learn', #'sklearn',
      'coloredlogs',
      'scikit-image',
      'h5py',
      'seaborn',
      'pyglet',
      'wandb>=0.12.21',
    ] + test_requirements,

    python_requires=">=3.6",
)

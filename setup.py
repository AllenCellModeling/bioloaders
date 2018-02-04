
m setuptools import setup
import sys

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()

setup(name='bioloaders',
      description='A repository of loading common datatypes in bio sciences into numpy (and other) arrays.',
      author='Johnson, Gregory R.',
      author_email='gregj@alleninstitute.org',
      url='https://github.com/AllenCellModeling/bioloaders',
      packages=['bioloaders'],
      install_requires=required)

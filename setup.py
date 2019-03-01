#!/usr/bin/env python

import os
from setuptools import setup

with open('hoomd_flowws/version.py') as version_file:
    exec(version_file.read())

setup(name='hoomd_flowws',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: BSD License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows using HOOMD-Blue',
      entry_points={
          'flowws_modules': [
              'DEMInteraction = hoomd_flowws:DEMInteraction',
              'Init = hoomd_flowws:Init',
              'Interaction = hoomd_flowws:Interaction',
              'Run = hoomd_flowws:Run',
              'RunHPMC = hoomd_flowws:RunHPMC',
              'ShapeDefinition = hoomd_flowws:ShapeDefinition',
          ],
      },
      extras_require={},
      install_requires=['flowws'],
      license='MIT',
      packages=[
          'hoomd_flowws',
      ],
      python_requires='>=3',
      version=__version__
      )

#!/usr/bin/env python

import os
from setuptools import setup

with open('hoomd_flowws/version.py') as version_file:
    exec(version_file.read())

module_names = [
    'Damasceno2017Interaction',
    'DEMInteraction',
    'Init',
    'Interaction',
    'Run',
    'RunHPMC',
    'ShapeDefinition',
]

flowws_modules = []
for name in module_names:
    flowws_modules.append('{0} = hoomd_flowws.{0}:{0}'.format(name))
    flowws_modules.append(
        'hoomd_flowws.{0} = hoomd_flowws.{0}:{0}'.format(name))

setup(name='hoomd_flowws',
      author='Matthew Spellings',
      author_email='matthew.p.spellings@gmail.com',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
      ],
      description='Stage-based scientific workflows using HOOMD-Blue',
      entry_points={
          'flowws_modules': flowws_modules,
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

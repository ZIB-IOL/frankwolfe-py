from setuptools import setup

setup(name='frankwolfepy',
      version='0.1.0',
      description='toolbox for Frank-Wolfe and conditional gradients algorithms in Python',
      packages=['frankwolfepy','frankwolfepy.tests'],
      install_requires=['juliacall>=0.9.14', 'jill'],
      include_package_data=True,
      zip_safe=False)
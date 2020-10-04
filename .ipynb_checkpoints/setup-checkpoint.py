from setuptools import setup, find_packages

setup(name='JSS',
      packages=find_packages(),
      version='1.0.0',
      install_requires=['gym', 'numpy', 'pandas', 'plotly']
)
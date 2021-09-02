from setuptools import find_packages, setup


setup(
    name='deafferentation-model',
    author='Brad Buran',
    author_email='bburan@alum.mit.edu',
    packages=find_packages(),
    include_package_data=True,
    description='Module for estimating deafferentation from evoked potentials',
)

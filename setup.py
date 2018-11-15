from setuptools import setup
from setuptools import find_packages
import os

data_dir = os.path.join('sawyer','vendor')
data_files = [(d, [os.path.join(d,f) for f in files]) for d, folders, files in os.walk(data_dir)]

with open("README.md", 'r') as f:
    long_description = f.read()

setup (
    name='sawyer',
    description='Environments for sawyer robot',
    license="MIT",
    long_description=long_description,
    packages = [
		package for package in find_packages()
	],
    install_requires = [
        'cached_property',
        'glfw',
        'gym',
        'mako',
        'mujoco-py<1.50.2,>=1.50.1',
        'numpy',
        'pytest', 
    ],
	data_files = data_files
)

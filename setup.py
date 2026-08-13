from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """Read requirements.txt and return a clean list of package names."""
    requirements = []
    with open(file_path) as f:
        requirements = f.readlines()
        requirements = [r.replace('\n', '') for r in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='ml_thermalproject',
    version='0.0.1',
    author='Reuben Abraham Solomon',
    author_email='solomonreuben1010@gmail.com',
    description='End-to-end ML + explainable AI framework for predicting thermoelectric figure of merit (zT),',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=get_requirements('requirements.txt'),
)
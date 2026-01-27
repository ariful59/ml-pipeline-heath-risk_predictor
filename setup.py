from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return list of requirements
    from requirements.txt file
    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

    return requirements


setup(
    name='mlproject_health_risk_prediction_classification',
    version='0.0.1',
    author='Ariful',
    author_email='arifulamin@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
   """"this function will return the list of required packages"""
   requirements=[]
   with open(file_path,'r')as file_obj:
      requirements=file_obj.readlines()
      requirements=[req.replace("\n","")for req in requirements]
      if HYPEN_E_DOT in requirements:
         requirements.remove(HYPEN_E_DOT)
      return requirements

setup(
    name='my-package',
    version='0.0.1',
    author='yasir',
    author_email='farooquiyasir140@gmail.com',
    packages=find_packages(),
    install_requires=['numpy','pandas','matplotlib','seaborn','scikit-learn'])

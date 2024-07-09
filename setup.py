

import setuptools
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepvog",
    version="1.2.0.dev1",
    author="Yuk-Hoi Yiu",
    author_email="yyhhoi@gmail.com",
    description="Deep VOG for gaze estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydsgz/DeepVOG",
    license="GNU General Public License v3.0",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        'Operating System :: OS Independent',
    ],
    # packages=find_packages(include=['deepvog', 'deepvog.*']),
    py_modules=["demo", "finetune"],
    package_data = {
        'deepvog':['model/*.py', 'model/*.h5'],
        #'model_weights':['model/*.h5'],
        
    },
    python_requires='>=3.9.0',
    install_requires=['numpy>=1.26.4',
                      'scikit-video>=1.1.11',
                      'scikit-image>=0.24.0',
                      'tensorflow>=2.16.0',
                      'keras>=3.4.0'],

    
)

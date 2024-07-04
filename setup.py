

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepvog",
    version="1.2.0",
    author="Yuk-Hoi Yiu et al.",
    author_email="Yuk Hoi Yiu",
    description="Deep VOG for gaze estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pydsgz/DeepVOG",
    license="GNU General Public License v3.0",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        "Programming Language :: Python :: 3.11.0",
        'Operating System :: OS Independent',
    ],
    package_data = {
        'deepvog':['model/*.py', 'model/*.h5'],
        #'model_weights':['model/*.h5'],
        
    },
    python_requires='>=3.11.0',
    install_requires=['numpy>=1.26.4',
                      'scikit-video>=1.1.11',
                      'scikit-image>=0.2    4.0',
                      'tensorflow-gpu>=2.16.2',
                      'keras>=3.4.1'],

    
)

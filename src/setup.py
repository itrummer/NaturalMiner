'''
Created on May 4, 2023

@author: immanueltrummer
'''
from setuptools import setup, find_packages

setup(
    name='naturalminer',
    version='0.1.0',    
    description='A data mining tool used via natural language commands',
    url='https://github.com/itrummer/NaturalMiner',
    author='Immanuel Trummer',
    author_email='immanuel.trummer@gmail.com',
    license='MIT License',
    packages=find_packages(
        where='src'
        ),
    install_requires=[
        'streamlit>=1.9, <=2.0',
        'psycopg2-binary>=2.8, <3',
        'jinja2>=3.0, <4',
        'transformers>=4.9, <5',
        'sentence-transformers>=2.0, <3',
        'psycopg2-binary>=2.9.6, <3',
        ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: GPU',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Topic :: Database :: Front-Ends',
        'Programming Language :: Python',
    ],
)
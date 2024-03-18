from distutils.core import setup
from setuptools import find_packages

with open('README.md', encoding='utf-8') as file:
    description = file.read()

setup(
    name='textencserve',
    version='0.0.1',
    packages=find_packages(),
    license='Copyright (c) 2023 Hieu Pham',
    zip_safe=True,
    description='Python text encoding APIs.',
    long_description=description,
    long_description_content_type='text/markdown',
    author='Hieu Pham',
    author_email='64821726+hieupth@users.noreply.github.com',
    url='https://gitlab.com/hieupth/textencserve',
    keywords=[],
    install_requires=['uvicorn', 'fastapi', 'torch', 'tokenizers', 'underthesea', 'pyvi', 'ovmsclient'],
    classifiers=[
      'Development Status :: 1 - Planning',
      'Intended Audience :: Developers',
      'Topic :: Software Development :: Build Tools',
      'Programming Language :: Python :: 3'
    ],
)
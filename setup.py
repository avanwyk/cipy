import cipy

from setuptools import setup
from setuptools import find_packages

setup_warnings = list()


def read_md(file):
    try:
        from pypandoc import convert
    except ImportError:
        setup_warnings.append(
            "warning: 'pypandoc' not found, could not convert Markdown to RST")
        import codecs
        return codecs.open(file, 'r', 'utf-8').read()
    else:
        return convert(file, 'rst')


setup(name='cipy',
      version=cipy.__revision__,
      description='Computational Intelligence algorithms in Python',
      long_description=read_md('README.md'),
      classifiers=[
          'Development Status :: 1 - Planning',
          'License :: OSI Approved :: Apache Software License',
          'Programming Language :: Python :: 3.5',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering :: Artificial Intelligence',
      ],
      keywords='computational intelligence pso',
      url='https://github.com/avanwyk/cipy',
      author='Andrich van Wyk',
      author_email='abvanwyk@gmail.com',
      license='Apache License 2.0',
      platforms = ["any"],
      packages=find_packages(),
      install_requires=[
            'numpy',
      ],
      include_package_data=True,
      zip_safe=False)

print("\n".join(setup_warnings))

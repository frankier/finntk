import setuptools

setuptools.setup(
    name="finntk",
    version="0.0.12",
    url="https://github.com/frankier/finntk",
    author="Frankie Robertson",
    description="Finnish NLP toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        "more_itertools>=4.1.0",
        "pyahocorasick>=1.1.8",
        "appdirs>=1.4.3",
        "plumbum>=1.6.6",
        "nltk>=3.3",
        "gensim>=3.4.0",
    ],
    extras_require={
        "docs": [
            "sphinx_autodoc_typehints",
            "sphinx>=1.5",
            "sphinx_rtd_theme",
            "pytest_check_links",
            "recommonmark",
        ],
        "dev": [
            "pytest",
            # Markdown descriptions
            "twine>=1.11.0",
            "wheel>=0.31.0",
            "setuptools>=38.6.0",
        ],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)

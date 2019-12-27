import setuptools

setuptools.setup(
    name="finntk",
    version="0.0.68",
    url="https://github.com/frankier/finntk",
    author="Frankie Robertson",
    description="Finnish NLP toolkit",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={"finntk": ["vendor/cnn_50_100_512_4096_sample.json"]},
    install_requires=[
        "more_itertools>=4.1.0",
        "pyahocorasick",
        "appdirs>=1.4.3",
        "plumbum>=1.6.6",
        "nltk>=3.3",
        "gensim>=3.4.0",
        # Dependency of gensim - earlier versions can resolution problems
        "smart_open>=1.7.1",
        "portalocker>=1.2.1",
        "scikit-learn>=0.22.0",
        "wordfreq>=2.2.0",
    ],
    extras_require={
        "docs": [
            "sphinx_autodoc_typehints",
            "sphinx>=1.5",
            "sphinx_rtd_theme",
            "recommonmark",
        ],
        "dev": [
            "pytest",
            "hypothesis",
            # Markdown descriptions
            "twine>=1.11.0",
            "wheel>=0.31.0",
            "setuptools>=38.6.0",
            "flake8>=3.5.0",
        ],
        "gpu": [
            # Dependency of pytorch_pretrained_bert
            "torch>=1.0.0",
            "pytorch_pretrained_bert",
        ],
        "conceptnet5": ["ConceptNet>=5.7.0"],
    },
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
)

import setuptools

setuptools.setup(
    name="finntk",
    version="0.0.2",
    url="https://github.com/frankier/finntk",

    author="Frankie Robertson",

    description="Finnish NLP toolkit",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),

    install_requires=[
        "more_itertools>=4.1.0"
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

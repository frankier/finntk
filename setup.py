import setuptools

setuptools.setup(
    name="finntk",
    version="0.0.1",
    url="https://github.com/frankier/finntk",

    author="Frankie Robertson",

    description="Finnish NLP toolkit",
    long_description=open('README.md').read(),

    packages=setuptools.find_packages(),

    install_requires=[],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
)

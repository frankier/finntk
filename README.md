# FinnTK

Some simple high level tools for processing Finnish text.

This project is according to my personal preferences but might be
helpful to others, particularly for exploratory coding. For larger projects you may prefer to use [OMorFi](https://github.com/flammie/omorfi) directly.

## Development ##

### Release process ###

1. Make a release commit in which the version is incremented in setup.py

2. Make a git tag of this commit with `git tag v$VERSION`

3. Upload to PyPI with `python setup.py sdist bdist_wheel` and `twine upload dist/*`

# FinnTK

<p align="center">
<a href="https://badge.fury.io/py/finntk"><img src="https://badge.fury.io/py/finntk.svg" alt="Available on PyPI" height="18"></a>
<a href="https://finntk.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/finntk/badge/?version=latest"></a>
</p>

Some simple high level tools for processing Finnish text.

This project is according to my personal preferences but might be
helpful to others, particularly for exploratory coding. For larger projects you may prefer to use [OMorFi](https://github.com/flammie/omorfi) directly.

## Installation ##

This project assumes you've installed HFST and OMorFi system-wide, like so:

  $ PIP_IGNORE_INSTALLED=1 pipenv install --site-packages finntk

Part of the reason for this is because [HFST is not currently pip installable](https://github.com/hfst/hfst/issues/375).

The current known good versions of HFST and OMorFi are in installed with Docker in the `docker` directory.

## License ##

This project is licensed under the Apache v2 license. The code in
`finntk/vendor` is vendorized from other projects and is licensed under their
respective licenses.

## Development ##

### Release process ###

1. Make a release commit in which the version is incremented in setup.py

2. Make a git tag of this commit with `git tag v$VERSION`

3. Push release to GitHub with `git push --tags`

4. Upload to PyPI with `python3 setup.py sdist bdist_wheel` and `twine upload dist/*`

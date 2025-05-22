## Installation

### OS X (zsh)

To install everything in developer mode,
 ```shell
 pip install -e '.[test,examples]'
 ```

### OS X (bash), Windows (cmd prompt)

To install everything in developer mode,
 ```shell
 pip install -e .[test,examples]
 ```

## Testing

### using tox (recommended)
If you don't already have tox installed, you can pip install it
```
pip install tox
```

To run unit tests, run
```
tox run
```


### using pytest
To run unit tests, use pytest:

```
 pytest --cov=pyest --cov-report term-missing tests
 ```

To run unit tests with performance benchmarking:
```
pytest --benchmark-save=benchmark --benchmark-compare --cov=pyest --cov-report term-missing tests
```

## Documentation

To build the documentation:

```
pip install sphinx sphinx-rtd-theme
```

Then,
```
cd docs
make html
```

If you are using Windows powershell and don't have `make` installed, you can
alternatively run
```
cd docs
sphinx-build -b html . _build
```
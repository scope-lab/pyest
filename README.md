# pyest

## Citing this work

If you use this package in your scholarly work, please cite the following articles:

[K.A. LeGrand and S. Ferrari, “Split Happens! Imprecise and Negative Information in Gaussian Mixture Random Finite Set Filtering,” Journal of Advances in Information Fusion,  Vol 17, No. 2, December, 2022](http://keithlegrand.com/wp/wp-content/uploads/2023/05/LeGrand-2022-Split-Happens-Imprecise-and-Negative-Information-in-Gaussian-Mixture-Random-Finite-Set-Filtering.pdf)

J. Kulik and K.A. LeGrand, “Nonlinearity and Uncertainty Informed Moment-Matching Gaussian Mixture Splitting,” https://arxiv.org/abs/2412.00343


```
@article{legrand2022SplitHappensImprecise,
  title = {Split {{Happens}}! {{Imprecise}} and {{Negative Information}} in {{Gaussian Mixture Random Finite Set Filtering}}},
  author = {LeGrand, Keith A. and Ferrari, Silvia},
  year = {2022},
  month = dec,
  journal = {Journal of Advances in Information Fusion},
  volume = {17},
  number = {2},
  eprint = {2207.11356},
  primaryclass = {cs, eess},
  pages = {78--96},
  doi = {10.48550/arXiv.2207.11356},
}
@misc{kulik2024NonlinearityUncertaintyInformed,
  title = {Nonlinearity and {{Uncertainty Informed Moment-Matching Gaussian Mixture Splitting}}},
  author = {Kulik, Jackson and LeGrand, Keith A.},
  year = {2024},
  month = nov,
  number = {arXiv:2412.00343},
  eprint = {2412.00343},
  primaryclass = {stat},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2412.00343},
  urldate = {2025-01-01},
  archiveprefix = {arXiv}
}

```

## Installation

### OS X (zsh)
To install, run
```shell
pip install -e .
```

To install packages needed for testing,
```shell
pip install -e '.[test]'
```
 To install packages needed for running the examples, run
 ```shell
 pip install -e '.[examples]'
 ```
 or to install everything,
 ```shell
 pip install -e '.[test,examples]'
 ```

### OS X (bash), Windows (cmd prompt)
To install, run
```shell
pip install -e .
```

To install packages needed for testing,
```shell
pip install -e .[test]
```
 To install packages needed for running the examples, run
 ```shell
 pip install -e .[examples]
 ```
 or to install everything,
 ```shell
 pip install -e .[test,examples]
 ```


## Testing

### using tox (recommended)
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

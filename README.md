# information

[![MIT licensed](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE.md)
![actions status](https://github.com/noamteyssier/information/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/noamteyssier/information/branch/main/graph/badge.svg?token=TL92OX90AM)](https://codecov.io/gh/noamteyssier/information)

ndarray-based information theory utilities.

## About

This is a crate to calculate information theory metrics with [ndarray](https://docs.rs/ndarray).

Check out the [docs](https://docs.rs/information) for usage and examples.

## Functions

This calculates `entropy`, `conditional_entropy`, `joint_entropy`,
`mutual_information`, and `conditional_mutual_information`.

## Utilities

All of the above functions expect probability matrices - but this crate exposes
some utility functions to build individual and joint probability densities for
multiple variables using the `prob*` and `hist*` functions.

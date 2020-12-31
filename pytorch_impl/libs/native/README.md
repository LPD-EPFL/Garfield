# Classification

* `so` -> "Vanilla" shared object: build and load
* `py` -> "Python" shared object: build, load and bind as submodule (e.g. 'py_test' with be available at 'native.test')

# Dependencies

In dependent SO directory, create a '.deps' file with new line-separated list of dependee SO directory.

# External dependencies

* `pip3 install ninja`
* `https://github.com/NVlabs/cub`

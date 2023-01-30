# ADME Prediction :microscope:

Description.


## Building the Environment :hammer:

All dependencies can be found in the `pyproject.toml` file. The setup instructions 
below assumes you have [pyenv](https://github.com/pyenv/pyenv) and 
[poetry](https://python-poetry.org/) installed. To set up the poetry environment, run 
the following in a shell:

```
$ poetry env use 3.10
$ poetry update
```

To activate the environment in a nested shell, run:

```
$ poetry shell
```


## Running the Jupyter Notebook :running:

The jupyter notebook has been converted to markdown using 
[jupytext](https://jupytext.readthedocs.io/en/latest/install.html) (to reduce storage 
requirements on GitHub). To convert it back to `.ipynb` format and run it in the nested 
shell, run:

```
$ jupytext --to ipynb *.md 
$ jupyter notebook
```

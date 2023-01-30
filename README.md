# ADME Prediction :microscope:

Description.


## Building the Environment :hammer:

To build the environment, run the following commands in a shell:

```
$ python -m venv env
$ source env/bin/activate
$ (env) source setup.sh
```

Note that only some of the dependencies are in the `requirements.txt` file; merely 
installing these is not sufficient. The reason the `setup.sh` script exists is because [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) needs to be 
installed from source and involves several steps which need to be executed in sequence, 
and one of its dependencies needs to be installed directly from GitHub as it not hosted 
on PyPi.

To create a Jupyter kernel, run:

```
$ (env) ipython kernel install --name "env" --user
```

**Note**: The setup instructions above have only been tested for python 3.10 on M1 Mac. 


## Running the Jupyter Notebook :running:

The [Jupyter](https://jupyter.org/) notebook has been converted to markdown using 
[Jupytext](https://jupytext.readthedocs.io/en/latest/install.html) (to reduce storage requirements on GitHub). To convert it back to `.ipynb` format and run it, run the following commands:

```
$ (env) jupytext --to ipynb *.md 
$ (env) jupyter notebook
```

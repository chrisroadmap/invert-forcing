# invert-forcing
Can we invert an energy balance model to estimate effective radiative forcing?

This is currently a work in progress, just hacking around with a couple of models and their EBM representations. Processed ESM data is from Gergana Gyuleva.

## to install

### set up environment

```
git clone git@github.com:chrisroadmap/invert-forcing.git
cd invert-forcing
conda env create -f environment.yml
conda activate invert-forcing
```

if you need to change the package lists in `environment.yml`:
```
conda env update -f environment.yml --prune
```

## running notebooks

```
jupyter notebook
cd notebooks
```

Note that I use `jupytext` and notebooks are committed as plain `python` files. The notebooks are not committed. If your environment is installed correctly, you should be able to open these files in `jupyter` and run them as normal.

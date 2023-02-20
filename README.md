# Complexity-based Dataset Generation

An Evolutionary Scalable Framework for Synthetic Data Generation based in Data Complexity. 

**🇬🇧 English** - [🇧🇷 Português Brasileiro](./README_pt-br.md)

`cbdgen` (**C**omplexity-**b**ased **D**ataset **Gen**eration) is a software, currently in development to become a framework, that implements a many-objective algorithm to generate synthetic datasets from characteristics (complexities).

## Requirements

Due to the actual state of the framework, a few steps are necessary/optional to run the framework. Here we list the requirements to run this project, as well as a few tutorials:

1. [Install R](./docs/setup-r.md)
2. Install Python
3. [Python Environment (Optional)](./docs/setup-python-venv.md)
4. [Setup `cbdgen`](#setting-up)

## Setting up

### Install R packages

It is required `ECoL` package to correctly calculate data complexity, to do so you can use the following command:

```console
./install_packages.r
```

> If you've successfully installed R, this Rscript will work fine, but if you get any error using the R environment, Try [Working with ECoL](./examples/ECoL-in-python.ipynb) notebook to setup `ECoL` package with Python.

### Install Python dependencies

Let's use `pip` to install our packages based on our `requirements.txt`.

```console
pip install --upgrade pip
pip install -r requirements.txt
```

Now you're ready to Generate Synthetic Data!

## Citation

```BibTeX
@inproceedings{Pereira_A_Many-Objective_Optimization_2022,
author = {Pereira, Steffano X. and Miranda, Péricles B. C. and França, Thiago R. F. and Bastos-Filho, Carmelo J. A. and Si, Tapas},
booktitle = {2022 IEEE Latin American Conference on Computational Intelligence (LA-CCI)},
doi = {10.1109/la-cci54402.2022.9981848},
month = {11},
pages = {1--6},
title = {{A Many-Objective Optimization Approach to Generate Synthetic Datasets based on Real-World Classification Problems}},
year = {2022}
}
```

For more details, see [CITATION.cff](./CITATION.cff).

## References

Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho, T. K. (2019). How Complex Is Your Classification Problem?: A Survey on Measuring Classification Complexity. ACM Computing Surveys (CSUR), 52:1-34.

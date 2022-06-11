# Setup R

`cbdgen` is a software that operates with various packages produced by the open-source community, one of these packages is the complexity package `ECoL`, which is responsible to analyze a complexity of a dataset. This is a package built in `R` and then it is necessary that our framework based in `python` must recognize this package. A way to do that is to use a conversion package from Python to R, in this case we use `rpy2` which can be installed using `pip`, but before installing using `pip`, we must have all the development kit from R Language, thus is necessary to install `r-base` and add it to your PATH.

> Disclaimer: This tutorial is based in Linux-Debian systems, such as ubuntu. In case if you are a Windows user, try with CRAN: <https://cran.r-project.org/bin/windows/base/>.

1. Update your package manager

    ```console
    sudo apt-get update
    ```

2. Install `r-base`

    ```console
    sudo apt -y install r-base
    ```

3. Verify if `r-base` is configured successfully in your PATH

    ```console
    user@Ubuntu:~$ R
    ```

    If a R environment is prompted, then you successfully configured `R` to your PATH.

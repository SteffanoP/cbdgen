# Create a Virtual Environment in Python (`venv`)

To go even deeper with the development of the framework, it is interesting to work with a virtual environment in Python. To achieve that, Python has the right tool: `venv`. `venv` is a virtual environment created to store specific packages for your project.

1. Create a new virtual environment `venv`

    Create a directory whereas the virtual environment will be located by creating Python `venv` command.

    ```console
    python3 -m venv /path/to/environment/venvCBDGEN
    ```

    In case you don't have Python virtual environment, you'll be prompted with a error message to install `virtualenv`. Install it by using `pip install virtualenv` and try again.

2. Access the virtual environment

    Verify if you can access your environment by using:

    ```console
    source /path/to/environment/venvCBDGEN/bin/activate
    ```

    If so, you must see that your console might be showing the name of the environment, as the example below:

    ```console
    (venvCBDGEN) steffanop@asus-b85:~/GitHub/cbdgen-framework$
    ```

In case this tutorial didn't work for you, try taking a look at the official documentation <https://docs.python.org/3/library/venv.html>.

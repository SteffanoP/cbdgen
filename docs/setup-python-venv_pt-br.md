# Crie um ambiente virtual Python (`venv`)

Para irmos mais a fundo no desenvolvimento do framework, é interessante se trabalhar com um ambiente de desenvolvimento em python. Para isso Python possuí a ferramenta perfeita para isso que é o ambiente `venv`, que é um ambiente virtual python para desenvolvimento.

1. Crie um novo ambiente virtual `venv`

    Para isso é necessário especificar um diretório onde você deseja criar o seu ambiente virtual

    ```console
    python3 -m venv /path/to/environment/venvCBDGEN
    ```

    Caso você ainda não tenha o Ambiente Virtual Python, você será notificado com uma mensagem de erro para instalar o `virtualenv`. Instale-o usando o comando `pip install virtualenv` e tente novamente.

2. Acesse o ambiente virtual

    Verifique se você consegue acessar seu ambiente virtual com:

    ```console
    source /path/to/environment/venvCBDGEN/bin/activate
    ```

    Caso você consiga, você deverá ver uma alteração no seu terminal como a seguinte:

    ```console
    (venvCBDGEN) steffanop@asus-b85:~/GitHub/cbdgen-framework$
    ```

Caso esse tutorial não tenha funcionado para você, tente olhar a documentação oficial <https://docs.python.org/3/library/venv.html>.

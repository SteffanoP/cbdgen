# cbdgen-framework

Um Framework evolucionário escalável para geração de dados sintéticos baseada em complexidade.

## Instalação

Para reproduzir o framework pela primeira vez é necessário reproduzir alguns passos para o completo funcionamento do programa.

### Setup R

O `cbdgen` é um software que opera com base em diversos pacotes produzidos pela comunidade acadêmica, um deles é o pacote de complexidade `ECoL`, do qual é responsável de analisar a complexidade de um dataset. Esse é um pacote nativo da linguagem de programação `R` e portanto é necessário que o nosso framework de base em `python` possa reconhecer esse pacote. Uma forma de fazer isso é utilizar um pacote de conversão de Python para R, é o caso do pacote `rpy2` que pode ser instalado nativamente por meio do `pip`, todavia antes de realizar sua instalação é necessário possuir as ferramentas da linguagem de programação `R`, logo é necessário instalar o `R` e adicionar a sua path.

1. Atualize o cache do gerenciador de pacotes

    ```bash
    sudo apt-get update
    ```

2. Instale o ambiente `R`

    ```bash
    sudo apt -y install r-base
    ```

3. Verifique a instalação do ambiente `R`

    ```bash
    user@Ubuntu:~$ R
    ```

Caso você seja um usuário Windows, tente por meio do CRAN em: <https://cran.r-project.org/bin/windows/base/>

Caso você deseja obter mais detalhes acerca do pacote `ECoL` em Python, este repositório possuí um Notebook que detalha o funcionamento em: [ECoL-in-python.ipynb](examples/ECoL-in-python.ipynb)

### (Opcional) Crie um ambiente virtual Python

Para irmos mais a fundo no desenvolvimento do framework, é interessante se trabalhar com um ambiente de desenvolvimento em python. Para isso Python possuí a ferramenta perfeita para isso que é o ambiente `venv`, que é um ambiente virtual python para desenvolvimento.

1. Crie um novo ambiente virtual `venv`

    Para isso é necessário especificar um diretório onde você deseja criar o seu ambiente virtual

    ```bash
    python3 -m venv /path/to/environment/venvCBDGEN
    ```

2. Acesse o ambiente virtual

    Verifique se você consegue acessar seu ambiente virtual com:

    ```bash
    source /path/to/environment/venvCBDGEN/bin/activate
    ```

    Caso você consiga, você deverá ver uma alteração no seu terminal como a seguinte:

    ```terminal
    (venvCBDGEN) steffanop@asus-b85:~/GitHub/cbdgen-framework$
    ```

### Instalação dos pacotes necessários

Python possuí um gerenciador de pacotes embutido, conhecido como `pip`, vamos utilizar esse gerenciador para que possamos instalar nossos pacotes, esse repositório possuí uma lista dos pacotes em `requirements.txt`, logo só basta usar o seguinte código:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Citação

Para citar o `cbdgen-framework` em suas publicações utilize:

```BibTeX
@inproceedings{Franca_A_Many-Objective_optimization_2020,
author = {França, Thiago R. and Miranda, Péricles B. C. and Prudêncio, Ricardo B. C. and Lorena, Ana C. and Nascimento, André C. A.},
booktitle = {2020 IEEE Congress on Evolutionary Computation (CEC)},
doi = {10.1109/CEC48606.2020.9185543},
month = {7},
pages = {1--8},
title = {{A Many-Objective optimization Approach for Complexity-based Data set Generation}},
year = {2020}
}
```

Para mais detalhes veja a [CITATION.cff](CITATION.cff).

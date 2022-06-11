# Setup R

O `cbdgen` é um software que opera com base em diversos pacotes produzidos pela comunidade acadêmica, um deles é o pacote de complexidade `ECoL`, do qual é responsável de analisar a complexidade de um dataset. Esse é um pacote nativo da linguagem de programação `R` e portanto é necessário que o nosso framework de base em `python` possa reconhecer esse pacote. Uma forma de fazer isso é utilizar um pacote de conversão de Python para R, é o caso do pacote `rpy2` que pode ser instalado nativamente por meio do `pip`, todavia antes de realizar sua instalação é necessário possuir as ferramentas da linguagem de programação `R`, logo é necessário instalar o `R` e adicionar a sua path.

> Aviso: Esse tutorial é baseado em sistemas Linux-Debian, como o ubuntu. Caso você seja um usuário Windows, tente por meio do CRAN em: <https://cran.r-project.org/bin/windows/base/>

1. Atualize o cache do gerenciador de pacotes

    ```console
    sudo apt-get update
    ```

2. Instale o ambiente `R`

    ```console
    sudo apt -y install r-base
    ```

3. Verifique a instalação do ambiente `R`

    ```console
    user@Ubuntu:~$ R
    ```

    Caso um ambiente R seja aberto, então você instalou o `R` corretamente ao seu PATH.

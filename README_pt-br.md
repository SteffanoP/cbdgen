# Complexity-based Dataset Generation

Um Framework evolucionário escalável para geração de dados sintéticos baseada em complexidade.

[🇬🇧 English](./README.md) - **🇧🇷 Português Brasileiro**

`cbdgen` (Complexity-based Dataset Generation) é um software, atualmente em desenvolvimento para se tornar um framework, que implementa um algoritmo para muitos objetivos que gera _datasets_ sintéticos baseado em características (complexidades de dados).

## Requisitos

Devido ao estado de desenvolvimento do framework, alguns passos são necessários/opcionais para o correto funcionamento do framework. Aqui listamos os requisitos para rodar o projeto, assim como alguns tutoriais:

1. [Install R](./docs/setup-r_pt-br.md)
2. Install Python
3. [Python Environment (Optional)](./docs/setup-python-venv_pt-br.md)
4. [Setup `cbdgen`](#instalação)

## Instalação

### Instalação dos pacotes `R`

O Pacote `ECoL` é necessário para calcular corretamente a complexidade dos dados, para fazer isso utilize o seguinte comando:

```console
./install_packages.r
```

> Se você instalou corretamente o ambiente `R`, esse `Rscript` irá funcionar corretamente, mas se você obter algum erro usando o ambiente `R`, tente [Working with ECoL](./examples/ECoL-in-python.ipynb) notebook para instalar o pacote `ECoL` com Python.

### Instalação das Dependências Python

Vamos usar o `pip` para instalar nossos pacotes baseado em nossos `requirements.txt`.

```console
pip install --upgrade pip
pip install -r requirements.txt
```

Agora você está pronto para gerar Datasets sintéticos!

## Citação

Para citar o `cbdgen-framework` em suas publicações utilize:

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

Para mais detalhes veja a [CITATION.cff](CITATION.cff).

## Referências

## References

Lorena, A. C., Garcia, L. P. F., Lehmann, J., Souto, M. C. P., and Ho, T. K. (2019). How Complex Is Your Classification Problem?: A Survey on Measuring Classification Complexity. ACM Computing Surveys (CSUR), 52:1-34.

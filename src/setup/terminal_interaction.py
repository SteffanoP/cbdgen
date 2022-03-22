def interact():
    maker = maker_type_input()
    samples = samples_input()
    attributes = attributes_input()
    classes = classes_input()

    filename = filename_input()

def maker_type_input() -> int:
    print("Escolha que tipo de base deseja gerar:")
    print("Escolha 1 - Para bolhas de pontos com uma distribuição gaussiana.")
    print("Escolha 2 - Para gerar um padrão de redemoinho, ou duas luas.")
    print("Escolha 3 - Para gerar um problema de classificação com conjuntos"
    "de dados em círculos concêntricos.")
    # TODO: Insert options 4 and 5
    maker_type = __input_with_default__("Opção 1 - 2 - 3:", 1, data_type=int)

    if(maker_type == 1): 
        m_option = __input_with_default__(
            "Quantas bolhas (centers) deseja utilizar?", 1, data_type=int)
    if (maker_type == 2):
        m_option = __input_with_default__(
            "Quanto de ruido deseja utilizar? entre 0 e 1", 0, data_type=float)
    if (maker_type == 3):
        m_option = __input_with_default__(
            "Quanto de ruido deseja utilizar? entre 0 e 1", 0, data_type=float)
    if (maker_type == 4):
        m_option = None
    if (maker_type == 5):
        m_option = __input_with_default__(
            "Quantas labels você deseja classificar?", 1, data_type=int)

    return maker_type, m_option

def samples_input() -> int:
    return __input_with_default__(
        "Quantas instancias (Exemplos) deseja utilizar?\n", 100, data_type=int)

def attributes_input() -> int:
    return __input_with_default__(
        "Quantos atributos (features) deseja utilizar?\n", 2, data_type=int)

def classes_input() -> int:
    return __input_with_default__(
        "Quantas classes você deseja?\n", 2, data_type=int)

def filename_input() -> int:
    return __input_with_default__(
        "Como deseja nomear o arquivo do dataset gerado?\n", "", data_type=str)


def __input_with_default__(input_text: str, default_value, data_type):
    try:
        data_type(input(input_text))
    except ValueError:
        print(f"Invalid Value, using default: {default_value}")
        return default_value

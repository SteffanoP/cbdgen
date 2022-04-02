from setup.options_types.cm import cm

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
    
def filepath_input() -> str:
    q = "Você deseja basear as métricas a um dataset já existente? (y/N)"
    if input(q) is 'y':
        return __input_with_default__("Digite o caminho para o dataset", "", 
                                      str)
    return ""

def measures_input() -> list:
    print("Escolha quais métricas e valores que deseja otimizar")
    print("Class imbalance C2")
    print("Linearity L2")
    print("Neighborhood N1")
    print("Neighborhood N2 (Experimental)")
    print("Neighborhood T1 (Experimental)")
    print("Feature-based F2")
    print("Ao digitar, obedeça o seguinte padrão:")
    print("<Medida de Complexidade>:<Valor de Complexidade Desejado [0,1]>")
    print("Por Exemplo: N1:0.5 L2:0.4 C2:0.35 F2:0.80\n")

    input_Stream = input().split()
    
    # Appends every Complexity Measure in measures list
    return [cm(measure) for measure in input_Stream]

def __input_with_default__(input_text: str, default_value, data_type):
    try:
        return data_type(input(input_text))
    except ValueError:
        print(f"Invalid Value, using default: {default_value}")
        return default_value

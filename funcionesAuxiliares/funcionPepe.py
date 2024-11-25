import random.randint as r


def IndicesTruncado(pequeno, actual):
    '''
    dado el tamano de la imagen mas pequeña y de
    la actual, escoge aleatoriamente elementos de
    la actual, tantos, uno por cada elemento de la pequeña
    '''
    indices = list(range(actual))
    while len(indices) > pequeno:
        del indices[r(0, len(indices)-1)]
    print(actual)
    return indices

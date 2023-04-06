import tomli


def load_toml(path):
    """
    return data like a dict (key value pair)
    :param path: file path
    :type path: str
    :return: tomi infos
    """

    with open(path, mode='rb') as f:
        return tomli.load(f)

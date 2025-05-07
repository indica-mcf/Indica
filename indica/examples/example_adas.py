from indica.configs.readers.adasconf import ADF11
from indica.readers import ADASReader


def example_adf11(
    element="h",
    **kwargs,
):
    adas_reader = ADASReader()

    scd = adas_reader.get_adf11("scd", element, ADF11[element]["scd"])
    acd = adas_reader.get_adf11("acd", element, ADF11[element]["acd"])
    return scd, acd

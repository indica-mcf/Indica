"""
Set of functions to work with PROV provenance
"""
import prov.model as prov

def get_prov_attribute(provenance:prov.ProvEntity, attr:str=""):
    try:
        attr_value = list(provenance.get_attribute(attr))
        attr_value = attr_value[0]

        return attr_value
    except IndexError:
        print(f"Attribute {attr} not present in provenance")
        return None
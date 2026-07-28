from .readers import JETConf
from .readers import ST40Conf
from .readers import MASTUConf

MACHINE_CONFS = {"st40": ST40Conf, "jet": JETConf, "mastu": MASTUConf}

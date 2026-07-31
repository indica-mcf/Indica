# Create database sql files

from st40_phys_trends.settings.conf_read_write import TREE_NAMES, GLOBAL_ALL, GLOBAL_SYSTEM_SPECIFIC
from indica.configs import ST40Conf
from copy import deepcopy
import os
import numpy as np

# Dir information to write .sql files in correct folder
ROOT_DIR = os.path.abspath(os.curdir)

# High-level tree type to collate e.g. all equilibrium reconstruction in the same table
CONF = ST40Conf()
INSTRUMENT_METHODS = CONF.INSTRUMENT_METHODS
_tree_types = []
for tree in TREE_NAMES:
    if tree in INSTRUMENT_METHODS:
        _tree_types.append(INSTRUMENT_METHODS[tree])
TREE_TYPES = np.unique(_tree_types)

DATABASE_DIR = f"{ROOT_DIR}/configs/"

def create_database():
    # BEST linkage
    sql_table = f"""DROP TABLE IF EXISTS `best_linkage`;

CREATE TABLE `best_linkage` (
    `pulseNo` int NOT NULL,"""
    for tree in TREE_NAMES:
        col_name = ""
        if tree in INSTRUMENT_METHODS:
            col_name += f"{INSTRUMENT_METHODS[tree]}_"
        col_name += f"{tree}_best"
        sql_col = f"""
        '{col_name}' varchar(25) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,"""
        sql_table += sql_col

    sql_options = f"""
    PRIMARY KEY (`pulseNo`),
    UNIQUE KEY `pulseNo_UNIQUE` (`pulseNo`),
    CONSTRAINT `equilibrium_efit_best_id_static` UNIQUE (`pulseNo`, `equilibrium_efit_best`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;"""
    sql_table += sql_options

    with open(f"{DATABASE_DIR}/best_linkage.sql", "w") as f:
        f.write(sql_table)

    # GLOBAL tables
    for _type in TREE_TYPES:
        table_name = f"{_type}_global"
        sql_table = f"""DROP TABLE IF EXISTS '{table_name}';

CREATE TABLE '{table_name}' (
    `pulseNo` int NOT NULL,
    `code_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `run_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `time` double NOT NULL,"""

        # List containing all the global quantities to be written to SQL
        quantities = deepcopy(GLOBAL_ALL)
        if _type in GLOBAL_SYSTEM_SPECIFIC:
            quantities += GLOBAL_SYSTEM_SPECIFIC[_type]

        for quantity in quantities:
            sql_col = f"""
    '{quantity}' double DEFAULT NULL,
    '{quantity}_error' double DEFAULT NULL,
    '{quantity}_stdev' double DEFAULT NULL,
    '{quantity}_d_dt' double DEFAULT NULL,"""
        
            sql_table += sql_col

        # Add final table options
        sql_options = f"""
    CONSTRAINT id PRIMARY KEY (`pulseNo`, `code_name`, `run_name`, `time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;"""
    
        sql_table += sql_options

        with open(f"{DATABASE_DIR}/{table_name}.sql", "w") as f:
            f.write(sql_table)

    # STATIC tables
    for _type in TREE_TYPES:
        table_name = f"{_type}_static"
        sql_table = f"""DROP TABLE IF EXISTS '{table_name}';

CREATE TABLE '{table_name}' (
    `pulseNo` int NOT NULL,
    `code_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `run_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `time` double NOT NULL,
    `data_quality_score` int DEFAULT -2,
    `datetime` datetime DEFAULT NULL,
    `git_hash` varchar(40) COLLATE utf8mb4_bin DEFAULT NULL,
    `version` varchar(45) COLLATE utf8mb4_bin DEFAULT NULL,
    CONSTRAINT id_static PRIMARY KEY (`pulseNo`, `code_name`, `run_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;"""

        with open(f"{DATABASE_DIR}/{table_name}.sql", "w") as f:
            f.write(sql_table)

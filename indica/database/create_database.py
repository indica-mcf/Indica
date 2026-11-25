"""Generate the SQL schema files used to create the trend database."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from indica.configs import ST40Conf
from indica.database.settings import READ_PARAMS
from indica.database.settings.conf_read_write import GLOBAL_ALL
from indica.database.settings.conf_read_write import GLOBAL_SYSTEM_SPECIFIC
from indica.database.settings.conf_read_write import TREE_NAMES

# Dir information to write .sql files in correct folder
ROOT_DIR = Path(__file__).parent.parent
CONF_DIR = f"{ROOT_DIR}/database/configs"


CONF = ST40Conf()
INSTRUMENT_METHODS = CONF.INSTRUMENT_METHODS
_tree_types = []
for tree in TREE_NAMES:
    if tree in INSTRUMENT_METHODS:
        _tree_types.append(INSTRUMENT_METHODS[tree])
TREE_TYPES = np.unique(_tree_types)


def create_database(database_name: str):
    """Create all SQL schema files used by the trend database."""
    _best_linkage()
    _global_table()
    _static_table()
    _workflow_table()
    _time_binning(database_name)

    return CONF_DIR


def _best_linkage():
    """Write the best-linkage table schema to disk."""
    table_name = "best_linkage"
    sql_table = f"""DROP TABLE IF EXISTS `{table_name}`;

CREATE TABLE `{table_name}` (
    `pulseNo` int NOT NULL,"""
    for tree in TREE_NAMES:
        col_name = ""
        if tree in INSTRUMENT_METHODS:
            col_name += f"{INSTRUMENT_METHODS[tree]}_"
        col_name += f"{tree}_best"
        sql_col = f"""
        `{col_name}` varchar(25) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin DEFAULT NULL,"""
        sql_table += sql_col

    sql_options = """
    PRIMARY KEY (`pulseNo`),
    UNIQUE KEY `pulseNo_UNIQUE` (`pulseNo`),
    CONSTRAINT `equilibrium_efit_best_id_static` UNIQUE (`pulseNo`, `equilibrium_efit_best`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
"""
    sql_table += sql_options

    with open(f"{CONF_DIR}/{table_name}.sql", "w+") as f:
        f.write(sql_table)


def _global_table():
    """Write the per-tree global-data table schemas."""
    for _type in TREE_TYPES:
        table_name = f"{_type}_global"
        sql_table = f"""DROP TABLE IF EXISTS `{table_name}`;

CREATE TABLE `{table_name}` (
    `pulseNo` int NOT NULL,
    `code_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `run_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `time` double NOT NULL,"""

        # List containing all the global quantities to be written to SQL
        quantities = deepcopy(GLOBAL_ALL)
        if _type in GLOBAL_SYSTEM_SPECIFIC:
            quantities += GLOBAL_SYSTEM_SPECIFIC[_type]

        for quantity in np.unique(quantities):
            sql_col = f"""
    `{quantity}` double DEFAULT NULL,
    `{quantity}_error` double DEFAULT NULL,
    `{quantity}_stdev` double DEFAULT NULL,
    `{quantity}_d_dt` double DEFAULT NULL,"""

            sql_table += sql_col

        # Add final table options
        sql_options = """
    CONSTRAINT id PRIMARY KEY (`pulseNo`, `code_name`, `run_name`, `time`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
"""

        sql_table += sql_options

        with open(f"{CONF_DIR}/{table_name}.sql", "w+") as f:
            f.write(sql_table)


def _static_table():
    """Write the per-tree static-data table schemas."""
    for _type in TREE_TYPES:
        table_name = f"{_type}_static"
        sql_table = f"""DROP TABLE IF EXISTS `{table_name}`;

CREATE TABLE `{table_name}` (
    `pulseNo` int NOT NULL,
    `code_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `run_name` varchar(12) COLLATE utf8mb4_bin NOT NULL,
    `data_quality_score` int DEFAULT -2,
    `datetime` DATETIME DEFAULT NULL,
    `git_id` varchar(40) COLLATE utf8mb4_bin DEFAULT NULL,
    `version` varchar(45) COLLATE utf8mb4_bin DEFAULT NULL,
    CONSTRAINT id_static PRIMARY KEY (`pulseNo`, `code_name`, `run_name`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
"""

        with open(f"{CONF_DIR}/{table_name}.sql", "w+") as f:
            f.write(sql_table)


def _workflow_table():
    """Write the workflow table schemas used to track run provenance."""

    for _type in TREE_TYPES:
        table_name = f"{_type}_workflow"
        sql_table = f"""DROP TABLE IF EXISTS `{table_name}`;

CREATE TABLE `{table_name}` (
    `id_workflow` INT NOT NULL AUTO_INCREMENT,
    `id_static` varchar(37) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
    `code_run` varchar(25) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin \
GENERATED ALWAYS AS (substring_index(`id_static`,_utf8mb4'@',-1)) STORED,
    `pulseNo` int GENERATED ALWAYS AS (cast(substring_index(\
`id_static`,_utf8mb4'@',1) as unsigned)) STORED,
    `parent_id_static` VARCHAR(37) CHARACTER SET utf8mb4 COLLATE \
utf8mb4_bin NOT NULL,
    `parent_code_run` varchar(25) CHARACTER SET utf8mb4 COLLATE \
utf8mb4_bin GENERATED ALWAYS AS (substring_index(`parent_id_static`,_utf8mb4'@',\
-1)) STORED,
    `parent_pulseNo` int GENERATED ALWAYS AS (cast(substring_index(\
`parent_id_static`,_utf8mb4'@',1) as unsigned)) STORED,
    `parent_usage` VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NULL,
    PRIMARY KEY (`id_workflow`))
ENGINE = InnoDB
DEFAULT CHARACTER SET = utf8mb4
COLLATE = utf8mb4_bin;
"""

        with open(f"{CONF_DIR}/{table_name}.sql", "w+") as f:
            f.write(sql_table)


def _time_binning(database_name: str):
    """Write the shared time-bin table and seed it with the common time labels."""
    table_name = "time_binning"
    sql_table = f"""DROP TABLE IF EXISTS `{table_name}`;

CREATE TABLE `time_binning` (
  `time_names` char(11) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL,
  `time` double DEFAULT NULL,
  PRIMARY KEY (`time_names`),
  UNIQUE KEY `time_names_UNIQUE` (`time_names`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_bin;
"""

    for t in READ_PARAMS["t_sql"]:
        _t_str = f"{float(t):.3f}"
        _t_words = f"{int(t * 1.0e3):6d}p00ms".replace(" ", "0").strip("")
        sql_col = f"""
INSERT INTO `{database_name}`.`{table_name}` (`time_names`, `time`) VALUES ('{_t_words}', '{_t_str}');"""
        sql_table += sql_col

    sql_table += "\n"

    with open(f"{CONF_DIR}/{table_name}.sql", "w+") as f:
        f.write(sql_table)

from typing import Iterable

from MDSplus import Connection

PLASMA_FLAG_NODE = r"\ST40::TOP.SUMMARY:PLASMA"


def plasma_flag(
    pulse: int,
    server: str = "smaug",
    tree: str = "ST40",
) -> float:
    conn = Connection(server)
    conn.openTree(tree, pulse)
    return float(conn.get(PLASMA_FLAG_NODE).data())


def has_plasma(
    pulse: int,
    server: str = "smaug",
    tree: str = "ST40",
) -> bool:
    return plasma_flag(pulse, server=server, tree=tree) == 1.0


def filter_pulses(
    pulses: Iterable[int],
    server: str = "smaug",
    tree: str = "ST40",
) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    skipped: list[int] = []
    for pulse in pulses:
        print(f"Checking pulse {pulse} for plasma")
        if has_plasma(pulse, server=server, tree=tree):
            valid.append(pulse)
        else:
            skipped.append(pulse)
    return valid, skipped

from dataclasses import dataclass, field
from typing import List


@dataclass
class VMCommand:
    """
    Single virtual machine instruction.
    """

    opcode: str
    data: bytes
    registers: List[int] = field(default_factory=list)


@dataclass
class VMState:
    """
    Initial VM state.
    """

    registers: List[bytes]
    stack: List[int] = field(default_factory=list)

from dataclasses import dataclass, field


@dataclass
class VMCommand:
    """
    Single virtual machine instruction.
    """

    opcode: str
    data: bytes
    registers: list[int] = field(default_factory=list)


@dataclass
class VMState:
    """
    Initial VM state.
    """

    registers: list[bytes]
    stack: list[int] = field(default_factory=list)

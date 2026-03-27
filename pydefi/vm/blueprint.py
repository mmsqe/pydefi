from dataclasses import dataclass, field
from typing import Any, List, Optional

from .vm_command import VMCommand, VMState


@dataclass
class Blueprint:
    """Blueprint model used by the VM-oriented eDSL."""

    from_token: Any
    amount_in: int
    receiver: str
    commands: List[VMCommand] = field(default_factory=list)
    initial_state: Optional[VMState] = None
    chain_id: Optional[int] = None

    def add_command(self, cmd: VMCommand) -> None:
        self.commands.append(cmd)

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_token": self.from_token,
            "amount_in": self.amount_in,
            "receiver": self.receiver,
            "commands": [c.__dict__ for c in self.commands],
            "initial_state": None
            if self.initial_state is None
            else {
                "registers": self.initial_state.registers,
                "stack": self.initial_state.stack,
            },
            "chain_id": self.chain_id,
        }

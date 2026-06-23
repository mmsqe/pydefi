"""Venom-backend bytecode tooling vendored from evm-smith.

- :mod:`pydefi.venom.balance_patch` — the balance-slot peephole patcher.
- :mod:`pydefi.venom.erc20_abi` — shared ERC-20 selectors / ABI encoders.
- :mod:`pydefi.venom.lean_oracle` — bridge to the EVMYulLean ``venom_run``
  reference oracle (the formal Venom semantics).
"""

from pydefi.venom import balance_patch, erc20_abi

__all__ = ["balance_patch", "erc20_abi"]

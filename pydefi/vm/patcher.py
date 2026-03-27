from typing import List


class Patcher:
    """Python helper for calldata patching operations."""

    @staticmethod
    def patch_calldata(template: bytes, offsets: List[int], dynamic_values: List[int]) -> bytes:
        """
        Replace uint256 placeholders at provided offsets.
        """
        if len(offsets) != len(dynamic_values):
            raise ValueError("offsets length must match dynamic_values length")

        # Mutable byte buffer mirrors in-place memory patching behavior.
        data = bytearray(template)

        for offset, value in zip(offsets, dynamic_values):
            if offset + 32 > len(data):
                raise ValueError(f"offset {offset} is out of calldata bounds")

            # Write uint256 as 32-byte big-endian value.
            value_bytes = value.to_bytes(32, byteorder="big")
            data[offset : offset + 32] = value_bytes

        return bytes(data)

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @notice Minimal interface for calling DeFiVM.execute.
interface IDeFiVM {
    function execute(bytes calldata program) external payable;
}

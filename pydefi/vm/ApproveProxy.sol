// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

interface IERC20 {
    function transferFrom(address from, address to, uint256 amount) external returns (bool);
}

interface IDeFiVM {
    function execute(bytes calldata program) external payable;
}

/**
 * @title ApproveProxy
 * @notice A proxy entry-point that lets users safely grant ERC-20 token
 *         allowances for DeFiVM programs without the security risks of
 *         approving the permissionless DeFiVM executor directly.
 *
 * Problem
 * -------
 * ``DeFiVM.execute()`` is permissionless — any caller can run any program.
 * Approving tokens directly to DeFiVM means *any* ``execute()`` call can
 * drain those approvals.
 *
 * Solution
 * --------
 * Users approve ERC-20 tokens to *this* proxy and invoke DeFiVM programs
 * through ``ApproveProxy.execute()`` instead of ``DeFiVM.execute()``.  The
 * proxy pulls the declared token deposits from the caller into DeFiVM *before*
 * forwarding the program.  Programs then operate on tokens already held by
 * DeFiVM — no in-program ``transferFrom`` back-channel is required.
 *
 * Because the proxy only calls ``token.transferFrom(msg.sender, vm, amount)``
 * (user → VM, caller-controlled), there is no shared mutable state and no
 * reentrancy risk.
 *
 * Usage
 * -----
 * 1. User: ``token.approve(approveProxy, amount)``  — once per token/session.
 * 2. User: ``approveProxy.execute{value: v}(program, deposits)``
 *           where ``deposits`` lists the tokens and amounts to pull into DeFiVM.
 * 3. DeFiVM program operates on the deposited tokens (e.g. calls
 *    ``token.transfer(recipient, amount)`` from the VM's balance).
 */
contract ApproveProxy {
    error DepositFailed();

    /// @dev The DeFiVM instance this proxy is paired with.
    address public immutable vm;

    /// @notice A single ERC-20 token deposit: token address + amount to pull
    ///         from the caller into DeFiVM before executing the program.
    struct Deposit {
        address token;
        uint256 amount;
    }

    /// @param _vm Address of the paired DeFiVM contract.
    constructor(address _vm) {
        require(_vm != address(0), "ApproveProxy: zero vm address");
        vm = _vm;
    }

    /**
     * @notice Deposit ERC-20 tokens into DeFiVM and then execute a program.
     *
     * For each entry in ``deposits``, this function calls
     * ``token.transferFrom(msg.sender, vm, amount)``, moving the tokens from
     * the caller directly into the paired DeFiVM contract.  It then forwards
     * the program (and any ETH) to ``DeFiVM.execute()``.
     *
     * @param program  Raw EVM bytecode to execute (forwarded to DeFiVM).
     * @param deposits List of ERC-20 tokens and amounts to deposit into DeFiVM
     *                 before program execution.  May be empty.
     */
    function execute(bytes calldata program, Deposit[] calldata deposits) external payable {
        address vmAddr = vm;
        IDeFiVM _vm = IDeFiVM(vmAddr);
        for (uint256 i = 0; i < deposits.length; i++) {
            uint256 amount = deposits[i].amount;
            if (amount == 0) continue;
            _safeTransferFrom(deposits[i].token, msg.sender, vmAddr, amount);
        }
        _vm.execute{value: msg.value}(program);
    }

    /**
     * @dev Call ``token.transferFrom(from, to, amount)`` and revert if it fails.
     *      Supports both standard ERC-20s that return ``bool`` and widely-used
     *      non-compliant tokens that return no value.
     */
    function _safeTransferFrom(address token, address from, address to, uint256 amount) internal {
        bool callOk;
        bool retOk;
        assembly {
            let ptr := mload(0x40)
            // transferFrom(address,address,uint256) selector
            mstore(ptr, shl(224, 0x23b872dd))
            mstore(add(ptr, 0x04), from)
            mstore(add(ptr, 0x24), to)
            mstore(add(ptr, 0x44), amount)

            // 4 + 32*3 = 100 bytes
            callOk := call(gas(), token, 0, ptr, 100, 0, 32)

            switch returndatasize()
            case 0 {
                // Non-standard ERC-20 (no return data): treat as success.
                retOk := 1
            }
            case 32 {
                // Standard ERC-20: require returned bool == true.
                returndatacopy(ptr, 0, 32)
                retOk := iszero(iszero(mload(ptr)))
            }
            default {
                // Any other return-data size is considered malformed.
                retOk := 0
            }
        }

        if (!(callOk && retOk)) revert DepositFailed();
    }
}

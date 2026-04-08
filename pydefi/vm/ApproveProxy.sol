// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ApproveProxy
 * @notice User-facing entrypoint for deposit pull + DeFiVM execution.
 *
 * Problem
 * -------
 * ``DeFiVM.execute()`` is permissionless — any caller can run any program.
 * If the VM executes in proxy storage context (``delegatecall``), then every
 * downstream call observes ``msg.sender == proxy`` and can consume arbitrary
 * token allowances users previously granted to the proxy.
 *
 * Solution
 * --------
 * Users approve tokens to this proxy and call ``ApproveProxy.execute()``.
 * The proxy pulls deposits from ``msg.sender`` directly into the paired VM and
 * then calls ``DeFiVM.execute(bytes)`` normally. This prevents VM programs
 * from spending approvals that were granted to the proxy by other users.
 *
 * Usage
 * -----
 * 1. User: ``token.approve(approveProxy, amount)``  — once per token/session.
 * 2. User: ``approveProxy.execute{value: v}(program, deposits)`` where
 *    ``deposits`` lists tokens and amounts to pull from the caller.
 * 3. Program executes in the VM contract context and spends assets held by VM.
 */
contract ApproveProxy {
    error DepositFailed();

    /// @dev The DeFiVM instance this proxy is paired with.
    address public immutable vm;

    /// @notice A single ERC-20 token deposit: token address + amount to pull
    ///         from the caller into VM before executing the program.
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
        * @notice Pull caller deposits into VM and execute VM program via call.
     *
     * For each entry in ``deposits``, this function calls
        * ``token.transferFrom(msg.sender, vm, amount)``. It then calls
        * ``vm.execute(bytes)`` on the paired DeFiVM contract.
     *
     * @param program  Raw VM bytecode to execute.
     * @param deposits List of ERC-20 tokens and amounts to pull into VM
     *                 before program execution. May be empty.
     */
    function execute(bytes calldata program, Deposit[] calldata deposits) external payable {
        for (uint256 i = 0; i < deposits.length; i++) {
            uint256 amount = deposits[i].amount;
            if (amount == 0) continue;
            _safeTransferFrom(deposits[i].token, msg.sender, vm, amount);
        }
        _callVm(abi.encodeWithSignature("execute(bytes)", program));
    }

    /// @notice Forward unknown selectors to VM fallback (e.g. DEX callbacks).
    fallback() external payable {
        _callVm(msg.data);
    }

    receive() external payable {}

    function _callVm(bytes memory data) internal {
        (bool ok, bytes memory ret) = vm.call{value: msg.value}(data);
        if (!ok) {
            assembly {
                revert(add(ret, 32), mload(ret))
            }
        }
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

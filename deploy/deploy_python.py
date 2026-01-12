"""
Deploy VInferenceAudit contract to Shardeum using Python/web3.py
Following OBLIVION's approach for Shardeum compatibility
"""
import os
import json
from web3 import Web3
from web3.middleware import ExtraDataToPOAMiddleware
from solcx import compile_source, install_solc

# Install solc if needed
try:
    install_solc('0.8.20')
except:
    pass

# Shardeum EVM Testnet Configuration
RPC_URL = "https://api-mezame.shardeum.org"
CHAIN_ID = 8119
PRIVATE_KEY = "e94eeecc753a37660a42995832aa9bfd283d8abe44446dfe6bd798a879aecff8"

# Contract source code
CONTRACT_SOURCE = '''
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract VInferenceAudit {
    string public constant NAME = "V-Inference Audit";
    string public constant VERSION = "1.0.0";
    
    address public owner;
    uint256 public totalAudits;
    
    struct Audit {
        bytes32 proofHash;
        string jobId;
        address auditor;
        uint256 timestamp;
        uint256 blockNumber;
        bool exists;
    }
    
    mapping(string => Audit) public audits;
    string[] public jobIds;
    
    event AuditAnchored(string indexed jobId, bytes32 proofHash, address indexed auditor, uint256 timestamp);
    
    constructor() {
        owner = msg.sender;
    }
    
    function anchorAudit(bytes32 proofHash, string memory jobId) external returns (bool success) {
        require(!audits[jobId].exists, "Audit already exists");
        require(proofHash != bytes32(0), "Invalid proof hash");
        
        audits[jobId] = Audit({
            proofHash: proofHash,
            jobId: jobId,
            auditor: msg.sender,
            timestamp: block.timestamp,
            blockNumber: block.number,
            exists: true
        });
        
        jobIds.push(jobId);
        totalAudits++;
        
        emit AuditAnchored(jobId, proofHash, msg.sender, block.timestamp);
        return true;
    }
    
    function verifyAudit(string memory jobId, bytes32 proofHash) external view returns (bool valid, bytes32 onChainHash) {
        Audit storage audit = audits[jobId];
        require(audit.exists, "Audit not found");
        onChainHash = audit.proofHash;
        valid = (proofHash == onChainHash);
        return (valid, onChainHash);
    }
    
    function auditExists(string memory jobId) external view returns (bool exists) {
        return audits[jobId].exists;
    }
    
    function getAudit(string memory jobId) external view returns (
        bytes32 proofHash, address auditor, uint256 timestamp, uint256 blockNumber, bool exists
    ) {
        Audit storage audit = audits[jobId];
        return (audit.proofHash, audit.auditor, audit.timestamp, audit.blockNumber, audit.exists);
    }
    
    function getAuditCount() external view returns (uint256 count) {
        return totalAudits;
    }
}
'''

def deploy():
    print("🚀 Deploying VInferenceAudit to Shardeum EVM Testnet")
    print("=" * 50)
    
    # Connect to Shardeum
    w3 = Web3(Web3.HTTPProvider(RPC_URL))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    
    if not w3.is_connected():
        print("❌ Failed to connect to Shardeum")
        return
    
    print(f"✅ Connected to Shardeum (Chain ID: {CHAIN_ID})")
    
    # Setup account
    account = w3.eth.account.from_key(PRIVATE_KEY)
    print(f"📝 Deploying with: {account.address}")
    
    balance = w3.eth.get_balance(account.address)
    balance_shm = w3.from_wei(balance, 'ether')
    print(f"💰 Balance: {float(balance_shm):.4f} SHM")
    print()
    
    # Compile contract
    print("📦 Compiling contract...")
    compiled = compile_source(CONTRACT_SOURCE, output_values=['abi', 'bin'], solc_version='0.8.20')
    contract_id, contract_interface = compiled.popitem()
    
    bytecode = contract_interface['bin']
    abi = contract_interface['abi']
    
    # Create contract instance
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Get gas price (using OBLIVION's 2x multiplier)
    gas_price = w3.eth.gas_price * 2
    print(f"⛽ Gas price: {w3.from_wei(gas_price, 'gwei'):.2f} gwei")
    
    # Build transaction
    nonce = w3.eth.get_transaction_count(account.address)
    
    tx = Contract.constructor().build_transaction({
        'from': account.address,
        'nonce': nonce,
        'gas': 1500000,
        'gasPrice': gas_price,
        'chainId': CHAIN_ID
    })
    
    # Estimate cost
    cost = gas_price * 1500000
    print(f"💵 Estimated cost: {float(w3.from_wei(cost, 'ether')):.4f} SHM")
    print()
    
    # Sign and send
    print("📤 Sending transaction...")
    signed_tx = w3.eth.account.sign_transaction(tx, PRIVATE_KEY)
    tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    print(f"📄 Transaction hash: {w3.to_hex(tx_hash)}")
    
    # Wait for receipt
    print("⏳ Waiting for confirmation...")
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash, timeout=180)
    
    if receipt['status'] == 1:
        contract_address = receipt['contractAddress']
        print()
        print("=" * 50)
        print("🎉 DEPLOYMENT SUCCESSFUL!")
        print("=" * 50)
        print()
        print(f"📋 Contract Address: {contract_address}")
        print()
        print("💡 Update backend/app/core/config.py with:")
        print(f'   CONTRACT_ADDRESS = "{contract_address}"')
        
        # Save ABI
        with open('VInferenceAudit_ABI.json', 'w') as f:
            json.dump(abi, f, indent=2)
        print("📄 ABI saved to VInferenceAudit_ABI.json")
        
        return contract_address
    else:
        print("❌ Transaction failed!")
        print(f"   Receipt: {receipt}")
        return None

if __name__ == "__main__":
    deploy()

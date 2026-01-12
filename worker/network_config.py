"""
V-OBLIVION Network Configuration
Shardeum EVM Testnet Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ==============================================
# SHARDEUM EVM TESTNET CONFIGURATION
# ==============================================

# Network Settings
RPC_URL = os.environ.get("RPC_URL", "https://api-mezame.shardeum.org")
CHAIN_ID = int(os.environ.get("CHAIN_ID", "8119"))
CHAIN_NAME = os.environ.get("CHAIN_NAME", "Shardeum EVM Testnet")
EXPLORER_URL = "https://explorer-mezame.shardeum.org"

# Contract Addresses
CONTRACT_ADDRESS = os.environ.get("CONTRACT_ADDRESS", "0xb3BD0a70eB7eAe91E6F23564d897C8098574e892")
ESCROW_CONTRACT_ADDRESS = os.environ.get("ESCROW_CONTRACT_ADDRESS", "0x0117A0EcF95dE28CCc0486D45D5362e020434575")

# Worker Settings
PRIVATE_KEY = os.environ.get("PRIVATE_KEY", "")
WORKER_TYPE = os.environ.get("WORKER_TYPE", "python")
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "2"))

# Differential Privacy Settings
ENABLE_DIFFERENTIAL_PRIVACY = os.environ.get("ENABLE_DIFFERENTIAL_PRIVACY", "true").lower() == "true"
PRIVACY_EPSILON = float(os.environ.get("PRIVACY_EPSILON", "1.0"))

# IPFS/Pinata Settings
PINATA_API_KEY = os.environ.get("PINATA_API_KEY", "")
PINATA_SECRET_KEY = os.environ.get("PINATA_SECRET_KEY", "")
IPFS_GATEWAY = "https://gateway.pinata.cloud/ipfs/"

# Staking Settings
MINIMUM_STAKE = 0.001  # SHM
STAKE_PERCENTAGE = 0.5  # 50% of reward

# Polling Settings
POLL_INTERVAL_IDLE = 5  # seconds when no jobs
POLL_INTERVAL_ACTIVE = 2  # seconds when processing
HEARTBEAT_INTERVAL = 15  # seconds

# Network Configuration Object
NETWORK_CONFIG = {
    "name": CHAIN_NAME,
    "chain_id": CHAIN_ID,
    "rpc_url": RPC_URL,
    "explorer": EXPLORER_URL,
    "native_currency": {
        "name": "Shardeum",
        "symbol": "SHM",
        "decimals": 18
    }
}

def get_explorer_tx_url(tx_hash: str) -> str:
    """Get explorer URL for a transaction"""
    return f"{EXPLORER_URL}/tx/{tx_hash}"

def get_explorer_address_url(address: str) -> str:
    """Get explorer URL for an address"""
    return f"{EXPLORER_URL}/address/{address}"

def validate_config():
    """Validate required configuration"""
    errors = []
    
    if not PRIVATE_KEY:
        errors.append("PRIVATE_KEY not set")
    
    if not RPC_URL:
        errors.append("RPC_URL not set")
    
    if not CONTRACT_ADDRESS:
        errors.append("CONTRACT_ADDRESS not set")
    
    return errors

if __name__ == "__main__":
    print(f"🔗 Network: {CHAIN_NAME}")
    print(f"⛓️ Chain ID: {CHAIN_ID}")
    print(f"🌐 RPC: {RPC_URL}")
    print(f"📜 Contract: {CONTRACT_ADDRESS}")
    print(f"🔐 DP Enabled: {ENABLE_DIFFERENTIAL_PRIVACY}")
    print(f"🔢 DP Epsilon: {PRIVACY_EPSILON}")
    
    errors = validate_config()
    if errors:
        print(f"\n⚠️ Configuration errors:")
        for e in errors:
            print(f"   - {e}")
    else:
        print(f"\n✅ Configuration valid")

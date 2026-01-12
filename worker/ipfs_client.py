"""
V-OBLIVION IPFS Client
Unified IPFS client supporting Pinata for decentralized storage
"""

import os
import json
import requests
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv()

# Pinata Configuration
PINATA_API_KEY = os.environ.get("PINATA_API_KEY", "")
PINATA_SECRET_KEY = os.environ.get("PINATA_SECRET_KEY", "")
PINATA_JWT = os.environ.get("PINATA_JWT", "")

# Gateway URLs
PINATA_GATEWAY = "https://gateway.pinata.cloud/ipfs/"
PUBLIC_GATEWAY = "https://ipfs.io/ipfs/"

# Pinata API endpoints
PINATA_PIN_FILE_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_PIN_JSON_URL = "https://api.pinata.cloud/pinning/pinJSONToIPFS"


class IPFSClient:
    """
    IPFS Client for V-OBLIVION
    Supports Pinata for reliable pinning
    """
    
    def __init__(self):
        self.api_key = PINATA_API_KEY
        self.secret_key = PINATA_SECRET_KEY
        self.jwt = PINATA_JWT
        
        # Check configuration
        self.is_configured = bool(self.api_key and self.secret_key) or bool(self.jwt)
        
        if self.is_configured:
            print("✅ IPFS client initialized (Pinata)")
        else:
            print("⚠️ IPFS: Pinata API keys not configured - using simulation mode")
    
    def _get_headers(self) -> Dict[str, str]:
        """Get authentication headers for Pinata"""
        if self.jwt:
            return {"Authorization": f"Bearer {self.jwt}"}
        return {
            "pinata_api_key": self.api_key,
            "pinata_secret_api_key": self.secret_key
        }
    
    def pin_file(self, file_path: str, name: Optional[str] = None) -> Optional[str]:
        """
        Pin a file to IPFS via Pinata
        Returns IPFS hash (CID) on success
        """
        if not self.is_configured:
            # Simulation mode
            import hashlib
            with open(file_path, 'rb') as f:
                content = f.read()
            fake_hash = "Qm" + hashlib.sha256(content).hexdigest()[:44]
            print(f"📤 [SIMULATED] Pinned file: {fake_hash}")
            return fake_hash
        
        try:
            with open(file_path, 'rb') as file:
                files = {"file": (name or os.path.basename(file_path), file)}
                
                metadata = {"name": name or os.path.basename(file_path)}
                options = {"cidVersion": 1}
                
                data = {
                    "pinataMetadata": json.dumps(metadata),
                    "pinataOptions": json.dumps(options)
                }
                
                response = requests.post(
                    PINATA_PIN_FILE_URL,
                    files=files,
                    data=data,
                    headers=self._get_headers()
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ipfs_hash = result.get("IpfsHash")
                    print(f"✅ Pinned to IPFS: {ipfs_hash}")
                    return ipfs_hash
                else:
                    print(f"❌ Pinata error: {response.status_code} - {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ IPFS pin error: {e}")
            return None
    
    def pin_bytes(self, data: bytes, name: str = "file") -> Optional[str]:
        """Pin raw bytes to IPFS"""
        if not self.is_configured:
            # Simulation mode
            import hashlib
            fake_hash = "Qm" + hashlib.sha256(data).hexdigest()[:44]
            print(f"📤 [SIMULATED] Pinned bytes: {fake_hash}")
            return fake_hash
        
        try:
            import io
            files = {"file": (name, io.BytesIO(data))}
            
            response = requests.post(
                PINATA_PIN_FILE_URL,
                files=files,
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result.get("IpfsHash")
                print(f"✅ Pinned bytes to IPFS: {ipfs_hash}")
                return ipfs_hash
            else:
                print(f"❌ Pinata error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ IPFS pin error: {e}")
            return None
    
    def pin_json(self, data: Dict[str, Any], name: str = "data.json") -> Optional[str]:
        """Pin JSON data to IPFS"""
        if not self.is_configured:
            # Simulation mode
            import hashlib
            json_str = json.dumps(data, sort_keys=True)
            fake_hash = "Qm" + hashlib.sha256(json_str.encode()).hexdigest()[:44]
            print(f"📤 [SIMULATED] Pinned JSON: {fake_hash}")
            return fake_hash
        
        try:
            payload = {
                "pinataContent": data,
                "pinataMetadata": {"name": name}
            }
            
            response = requests.post(
                PINATA_PIN_JSON_URL,
                json=payload,
                headers={**self._get_headers(), "Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                result = response.json()
                ipfs_hash = result.get("IpfsHash")
                print(f"✅ Pinned JSON to IPFS: {ipfs_hash}")
                return ipfs_hash
            else:
                print(f"❌ Pinata error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ IPFS pin error: {e}")
            return None
    
    def get_file(self, ipfs_hash: str) -> Optional[bytes]:
        """Download file from IPFS"""
        try:
            # Try Pinata gateway first
            url = f"{PINATA_GATEWAY}{ipfs_hash}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.content
            
            # Fallback to public gateway
            url = f"{PUBLIC_GATEWAY}{ipfs_hash}"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.content
            
            print(f"❌ Failed to fetch from IPFS: {ipfs_hash}")
            return None
            
        except Exception as e:
            print(f"❌ IPFS get error: {e}")
            return None
    
    def get_json(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """Download and parse JSON from IPFS"""
        content = self.get_file(ipfs_hash)
        if content:
            try:
                return json.loads(content.decode('utf-8'))
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse error: {e}")
        return None
    
    def get_url(self, ipfs_hash: str) -> str:
        """Get gateway URL for an IPFS hash"""
        return f"{PINATA_GATEWAY}{ipfs_hash}"


# Global client instance
_client: Optional[IPFSClient] = None

def get_ipfs_client() -> IPFSClient:
    """Get or create singleton IPFS client"""
    global _client
    if _client is None:
        _client = IPFSClient()
    return _client


# ============ CLI Functions ============

def main():
    """Test the IPFS client"""
    print("=" * 60)
    print("   IPFS CLIENT TEST")
    print("=" * 60)
    
    client = get_ipfs_client()
    
    # Test JSON pinning
    test_data = {
        "name": "V-OBLIVION Test",
        "timestamp": "2026-01-13",
        "version": "1.0.0"
    }
    
    print("\n📤 Testing JSON pin...")
    hash_result = client.pin_json(test_data, "test.json")
    
    if hash_result:
        print(f"   Hash: {hash_result}")
        print(f"   URL: {client.get_url(hash_result)}")
        
        # Test retrieval
        print("\n📥 Testing retrieval...")
        retrieved = client.get_json(hash_result)
        if retrieved:
            print(f"   Retrieved: {retrieved}")
        else:
            print("   ⚠️ Could not retrieve (may be propagating)")
    
    print("\n" + "=" * 60)
    print("✅ IPFS client test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

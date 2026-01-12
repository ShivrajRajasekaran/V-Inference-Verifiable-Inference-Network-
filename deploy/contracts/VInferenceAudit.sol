// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title VInferenceAudit
 * @dev Stores ZKML proof hashes on-chain for verification
 */
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
    
    event AuditAnchored(
        string indexed jobId,
        bytes32 proofHash,
        address indexed auditor,
        uint256 timestamp
    );
    
    event AuditVerified(
        string indexed jobId,
        bytes32 proofHash,
        bool valid,
        address indexed verifier
    );
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Not owner");
        _;
    }
    
    /**
     * @dev Anchor a proof hash on-chain
     * @param proofHash The hash of the ZK proof
     * @param jobId Unique job identifier
     */
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
    
    /**
     * @dev Verify a proof hash against on-chain record
     */
    function verifyAudit(string memory jobId, bytes32 proofHash) external returns (bool valid, bytes32 onChainHash) {
        Audit storage audit = audits[jobId];
        require(audit.exists, "Audit not found");
        
        onChainHash = audit.proofHash;
        valid = (proofHash == onChainHash);
        
        emit AuditVerified(jobId, proofHash, valid, msg.sender);
        return (valid, onChainHash);
    }
    
    /**
     * @dev Check if audit exists
     */
    function auditExists(string memory jobId) external view returns (bool exists) {
        return audits[jobId].exists;
    }
    
    /**
     * @dev Get audit details
     */
    function getAudit(string memory jobId) external view returns (
        bytes32 proofHash,
        address auditor,
        uint256 timestamp,
        uint256 blockNumber,
        bool exists
    ) {
        Audit storage audit = audits[jobId];
        return (
            audit.proofHash,
            audit.auditor,
            audit.timestamp,
            audit.blockNumber,
            audit.exists
        );
    }
    
    /**
     * @dev Get audit count
     */
    function getAuditCount() external view returns (uint256 count) {
        return totalAudits;
    }
    
    /**
     * @dev Get job ID by index
     */
    function getJobIdByIndex(uint256 index) external view returns (string memory jobId) {
        require(index < jobIds.length, "Index out of bounds");
        return jobIds[index];
    }
    
    /**
     * @dev Get recent audits
     */
    function getRecentAudits(uint256 count) external view returns (string[] memory recentJobIds) {
        uint256 total = jobIds.length;
        uint256 returnCount = count > total ? total : count;
        
        recentJobIds = new string[](returnCount);
        for (uint256 i = 0; i < returnCount; i++) {
            recentJobIds[i] = jobIds[total - 1 - i];
        }
        return recentJobIds;
    }
    
    /**
     * @dev Transfer ownership
     */
    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "Invalid address");
        owner = newOwner;
    }
}

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

// ============================================================================
// V-INFERENCE SMART CONTRACTS - REMIX IDE COMPATIBLE
// ============================================================================
// This file contains all V-Inference contracts flattened for Remix IDE.
// Deploy order: 1) ZKMLVerifier, 2) NodeRegistry, 3) InferenceMarketplace, 4) InferenceEscrow
// ============================================================================

// ============================================================================
// OpenZeppelin Dependencies (Flattened)
// ============================================================================

abstract contract Context {
    function _msgSender() internal view virtual returns (address) {
        return msg.sender;
    }

    function _msgData() internal view virtual returns (bytes calldata) {
        return msg.data;
    }
}

abstract contract Ownable is Context {
    address private _owner;

    error OwnableUnauthorizedAccount(address account);
    error OwnableInvalidOwner(address owner);

    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);

    constructor(address initialOwner) {
        if (initialOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(initialOwner);
    }

    modifier onlyOwner() {
        _checkOwner();
        _;
    }

    function owner() public view virtual returns (address) {
        return _owner;
    }

    function _checkOwner() internal view virtual {
        if (owner() != _msgSender()) {
            revert OwnableUnauthorizedAccount(_msgSender());
        }
    }

    function renounceOwnership() public virtual onlyOwner {
        _transferOwnership(address(0));
    }

    function transferOwnership(address newOwner) public virtual onlyOwner {
        if (newOwner == address(0)) {
            revert OwnableInvalidOwner(address(0));
        }
        _transferOwnership(newOwner);
    }

    function _transferOwnership(address newOwner) internal virtual {
        address oldOwner = _owner;
        _owner = newOwner;
        emit OwnershipTransferred(oldOwner, newOwner);
    }
}

abstract contract ReentrancyGuard {
    uint256 private constant NOT_ENTERED = 1;
    uint256 private constant ENTERED = 2;

    uint256 private _status;

    error ReentrancyGuardReentrantCall();

    constructor() {
        _status = NOT_ENTERED;
    }

    modifier nonReentrant() {
        _nonReentrantBefore();
        _;
        _nonReentrantAfter();
    }

    function _nonReentrantBefore() private {
        if (_status == ENTERED) {
            revert ReentrancyGuardReentrantCall();
        }
        _status = ENTERED;
    }

    function _nonReentrantAfter() private {
        _status = NOT_ENTERED;
    }

    function _reentrancyGuardEntered() internal view returns (bool) {
        return _status == ENTERED;
    }
}

interface IERC20 {
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);

    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 value) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 value) external returns (bool);
    function transferFrom(address from, address to, uint256 value) external returns (bool);
}

// ============================================================================
// CONTRACT 1: ZKMLVerifier
// Verifies EZKL-generated SNARK proofs for AI inference execution
// ============================================================================

contract ZKMLVerifier is Ownable, ReentrancyGuard {
    // Proof verification key (from EZKL circuit compilation)
    bytes public verificationKey;

    // Mapping of proof hash to verification status
    mapping(bytes32 => bool) public proofVerified;

    // Mapping of proof hash to inference job ID
    mapping(bytes32 => uint256) public proofToJobId;

    // Events
    event ProofVerified(bytes32 indexed proofHash, uint256 indexed jobId, address indexed prover);
    event ProofRejected(bytes32 indexed proofHash, uint256 indexed jobId, string reason);
    event VerificationKeyUpdated(bytes newKey);

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Verify a SNARK proof for inference execution
     * @param _proof The EZKL-generated proof
     * @param _publicInputs Public inputs to the circuit
     * @param _jobId The inference job ID
     * @return isValid Whether the proof is valid
     */
    function verifyProof(
        bytes calldata _proof,
        bytes calldata _publicInputs,
        uint256 _jobId
    ) external nonReentrant returns (bool isValid) {
        bytes32 proofHash = keccak256(_proof);

        // Prevent double-verification
        require(!proofVerified[proofHash], "Proof already verified");

        // Verify the SNARK proof (simplified - integrate actual EZKL verifier)
        isValid = _verifySnark(_proof, _publicInputs);

        if (isValid) {
            proofVerified[proofHash] = true;
            proofToJobId[proofHash] = _jobId;
            emit ProofVerified(proofHash, _jobId, msg.sender);
        } else {
            emit ProofRejected(proofHash, _jobId, "Invalid proof");
        }

        return isValid;
    }

    /**
     * @dev Internal SNARK verification logic
     * In production, integrate with EZKL's verifier contract
     */
    function _verifySnark(
        bytes calldata _proof,
        bytes calldata _publicInputs
    ) internal pure returns (bool) {
        // Placeholder: In production, call EZKL verifier
        // This validates the cryptographic proof
        return _proof.length > 0 && _publicInputs.length > 0;
    }

    /**
     * @dev Update the verification key (admin only)
     */
    function setVerificationKey(bytes calldata _newKey) external onlyOwner {
        verificationKey = _newKey;
        emit VerificationKeyUpdated(_newKey);
    }

    /**
     * @dev Check if a proof has been verified
     */
    function isProofVerified(bytes32 _proofHash) external view returns (bool) {
        return proofVerified[_proofHash];
    }
}

// ============================================================================
// CONTRACT 2: NodeRegistry
// Tracks inference node registrations and status
// ============================================================================

contract NodeRegistry is Ownable {
    struct Node {
        address nodeAddress;
        string hardwareType;
        uint256 registrationTime;
        bool isActive;
        uint256 reputation;
        uint256 jobsCompleted;
    }

    mapping(address => Node) public nodes;
    address[] public nodeList;

    event NodeRegistered(address indexed nodeAddress, string hardwareType);
    event NodeDeactivated(address indexed nodeAddress);
    event NodeReputationUpdated(address indexed nodeAddress, uint256 newReputation);

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Register a new inference node
     */
    function registerNode(string memory _hardwareType) external {
        require(!nodes[msg.sender].isActive, "Node already registered");

        nodes[msg.sender] = Node({
            nodeAddress: msg.sender,
            hardwareType: _hardwareType,
            registrationTime: block.timestamp,
            isActive: true,
            reputation: 100,
            jobsCompleted: 0
        });

        nodeList.push(msg.sender);
        emit NodeRegistered(msg.sender, _hardwareType);
    }

    /**
     * @dev Deactivate a node (admin only)
     */
    function deactivateNode(address _nodeAddress) external onlyOwner {
        require(nodes[_nodeAddress].isActive, "Node not active");
        nodes[_nodeAddress].isActive = false;
        emit NodeDeactivated(_nodeAddress);
    }

    /**
     * @dev Update node reputation after job completion
     */
    function updateReputation(address _nodeAddress, uint256 _delta, bool _increase) external onlyOwner {
        require(nodes[_nodeAddress].isActive, "Node not active");
        
        if (_increase) {
            nodes[_nodeAddress].reputation += _delta;
        } else {
            if (nodes[_nodeAddress].reputation > _delta) {
                nodes[_nodeAddress].reputation -= _delta;
            } else {
                nodes[_nodeAddress].reputation = 0;
            }
        }
        
        emit NodeReputationUpdated(_nodeAddress, nodes[_nodeAddress].reputation);
    }

    /**
     * @dev Increment jobs completed for a node
     */
    function incrementJobsCompleted(address _nodeAddress) external onlyOwner {
        nodes[_nodeAddress].jobsCompleted++;
    }

    /**
     * @dev Get node details
     */
    function getNode(address _nodeAddress) external view returns (Node memory) {
        return nodes[_nodeAddress];
    }

    /**
     * @dev Get total registered node count
     */
    function getNodeCount() external view returns (uint256) {
        return nodeList.length;
    }

    /**
     * @dev Get all active nodes
     */
    function getActiveNodes() external view returns (address[] memory) {
        uint256 activeCount = 0;
        for (uint256 i = 0; i < nodeList.length; i++) {
            if (nodes[nodeList[i]].isActive) {
                activeCount++;
            }
        }
        
        address[] memory activeNodes = new address[](activeCount);
        uint256 index = 0;
        for (uint256 i = 0; i < nodeList.length; i++) {
            if (nodes[nodeList[i]].isActive) {
                activeNodes[index] = nodeList[i];
                index++;
            }
        }
        
        return activeNodes;
    }
}

// ============================================================================
// CONTRACT 3: InferenceMarketplace
// Manages job posting, assignment, and completion
// ============================================================================

contract InferenceMarketplace is Ownable, ReentrancyGuard {
    enum HardwareType {
        CPU,
        GPU_LOW,
        GPU_HIGH,
        TPU
    }

    enum JobStatus {
        OPEN,
        ASSIGNED,
        COMPLETED,
        FAILED,
        CANCELLED
    }

    struct MarketplaceNode {
        address nodeAddress;
        HardwareType hardwareType;
        uint256 pricePerInference;
        bool isActive;
        uint256 lastHeartbeat;
        uint256 jobsCompleted;
    }

    struct InferenceJob {
        uint256 jobId;
        address requester;
        string modelCID;
        bytes inputData;
        HardwareType requiredHardware;
        uint256 budget;
        address assignedNode;
        JobStatus status;
        uint256 createdAt;
        uint256 completedAt;
    }

    mapping(address => MarketplaceNode) public marketplaceNodes;
    mapping(uint256 => InferenceJob) public jobs;
    mapping(address => bool) public registeredMarketplaceNodes;

    uint256 public jobCounter;
    uint256 public constant HEARTBEAT_TIMEOUT = 5 minutes;

    event MarketplaceNodeRegistered(address indexed nodeAddress, HardwareType hardwareType, uint256 price);
    event JobPosted(uint256 indexed jobId, address indexed requester, HardwareType hardwareType);
    event JobAssigned(uint256 indexed jobId, address indexed node);
    event JobCompleted(uint256 indexed jobId, address indexed node);
    event JobFailed(uint256 indexed jobId, string reason);
    event HeartbeatReceived(address indexed nodeAddress);

    constructor() Ownable(msg.sender) {}

    /**
     * @dev Register a new inference node in marketplace
     */
    function registerMarketplaceNode(HardwareType _hardwareType, uint256 _pricePerInference) public {
        require(_pricePerInference > 0, "Price must be greater than 0");

        marketplaceNodes[msg.sender] = MarketplaceNode({
            nodeAddress: msg.sender,
            hardwareType: _hardwareType,
            pricePerInference: _pricePerInference,
            isActive: true,
            lastHeartbeat: block.timestamp,
            jobsCompleted: 0
        });

        registeredMarketplaceNodes[msg.sender] = true;
        emit MarketplaceNodeRegistered(msg.sender, _hardwareType, _pricePerInference);
    }

    /**
     * @dev Post a new inference job
     */
    function postJob(
        string memory _modelCID,
        bytes memory _inputData,
        HardwareType _requiredHardware,
        uint256 _budget
    ) public returns (uint256) {
        require(_budget > 0, "Budget must be greater than 0");
        require(bytes(_modelCID).length > 0, "Model CID cannot be empty");

        uint256 jobId = jobCounter++;

        jobs[jobId] = InferenceJob({
            jobId: jobId,
            requester: msg.sender,
            modelCID: _modelCID,
            inputData: _inputData,
            requiredHardware: _requiredHardware,
            budget: _budget,
            assignedNode: address(0),
            status: JobStatus.OPEN,
            createdAt: block.timestamp,
            completedAt: 0
        });

        emit JobPosted(jobId, msg.sender, _requiredHardware);
        return jobId;
    }

    /**
     * @dev Assign a job to a node (self-assignment)
     */
    function assignJobToNode(uint256 _jobId) public {
        require(_jobId < jobCounter, "Job does not exist");
        InferenceJob storage job = jobs[_jobId];
        require(job.status == JobStatus.OPEN, "Job is not available");
        require(registeredMarketplaceNodes[msg.sender], "Node is not registered");
        require(marketplaceNodes[msg.sender].isActive, "Node is not active");
        require(marketplaceNodes[msg.sender].hardwareType == job.requiredHardware, "Hardware mismatch");
        require(marketplaceNodes[msg.sender].pricePerInference <= job.budget, "Price exceeds budget");

        job.assignedNode = msg.sender;
        job.status = JobStatus.ASSIGNED;

        emit JobAssigned(_jobId, msg.sender);
    }

    /**
     * @dev Mark job as completed
     */
    function completeJob(uint256 _jobId) external {
        require(_jobId < jobCounter, "Job does not exist");
        InferenceJob storage job = jobs[_jobId];
        require(job.status == JobStatus.ASSIGNED, "Job not assigned");
        require(job.assignedNode == msg.sender, "Not assigned node");

        job.status = JobStatus.COMPLETED;
        job.completedAt = block.timestamp;
        marketplaceNodes[msg.sender].jobsCompleted++;

        emit JobCompleted(_jobId, msg.sender);
    }

    /**
     * @dev Mark job as failed
     */
    function failJob(uint256 _jobId, string calldata _reason) external {
        require(_jobId < jobCounter, "Job does not exist");
        InferenceJob storage job = jobs[_jobId];
        require(job.status == JobStatus.ASSIGNED, "Job not assigned");
        require(job.assignedNode == msg.sender || msg.sender == owner(), "Not authorized");

        job.status = JobStatus.FAILED;
        emit JobFailed(_jobId, _reason);
    }

    /**
     * @dev Send heartbeat to maintain active status
     */
    function heartbeat() public {
        require(registeredMarketplaceNodes[msg.sender], "Node not registered");

        marketplaceNodes[msg.sender].lastHeartbeat = block.timestamp;
        marketplaceNodes[msg.sender].isActive = true;

        emit HeartbeatReceived(msg.sender);
    }

    /**
     * @dev Check if a node is alive
     */
    function isNodeAlive(address _nodeAddress) public view returns (bool) {
        if (!registeredMarketplaceNodes[_nodeAddress]) {
            return false;
        }
        return (block.timestamp - marketplaceNodes[_nodeAddress].lastHeartbeat) <= HEARTBEAT_TIMEOUT;
    }

    /**
     * @dev Get node details
     */
    function getMarketplaceNode(address _nodeAddress) public view returns (MarketplaceNode memory) {
        require(registeredMarketplaceNodes[_nodeAddress], "Node not registered");
        return marketplaceNodes[_nodeAddress];
    }

    /**
     * @dev Get job details
     */
    function getJob(uint256 _jobId) public view returns (InferenceJob memory) {
        require(_jobId < jobCounter, "Job does not exist");
        return jobs[_jobId];
    }

    /**
     * @dev Get total job count
     */
    function getJobCount() external view returns (uint256) {
        return jobCounter;
    }
}

// ============================================================================
// CONTRACT 4: InferenceEscrow
// Manages payment escrow with ZK proof-based release
// ============================================================================

interface IZKMLVerifier {
    function isProofVerified(bytes32 _proofHash) external view returns (bool);
}

contract InferenceEscrow is ReentrancyGuard, Ownable {
    IERC20 public paymentToken;
    IZKMLVerifier public verifier;

    uint256 public platformFeePercent = 2; // 2% platform fee
    address public feeRecipient;
    uint256 public accumulatedFees;

    enum EscrowJobStatus { PENDING, COMPLETED, DISPUTED, REFUNDED }

    struct EscrowJob {
        address requester;
        address provider;
        uint256 amount;
        EscrowJobStatus status;
        bytes32 proofHash;
        uint256 createdAt;
        uint256 completedAt;
    }

    mapping(uint256 => EscrowJob) public escrowJobs;
    uint256 public escrowJobCounter;

    // Native ETH escrow for those who prefer ETH
    mapping(uint256 => uint256) public ethEscrow;

    // Events
    event EscrowJobCreated(uint256 indexed jobId, address indexed requester, address indexed provider, uint256 amount);
    event EscrowJobCompleted(uint256 indexed jobId, bytes32 indexed proofHash, uint256 providerPayout);
    event EscrowJobDisputed(uint256 indexed jobId, string reason);
    event EscrowJobRefunded(uint256 indexed jobId, uint256 amount);
    event FeeWithdrawn(address indexed recipient, uint256 amount);
    event ETHDeposited(uint256 indexed jobId, uint256 amount);

    constructor(address _paymentToken, address _verifier, address _feeRecipient) Ownable(msg.sender) {
        paymentToken = IERC20(_paymentToken);
        verifier = IZKMLVerifier(_verifier);
        feeRecipient = _feeRecipient;
    }

    /**
     * @dev Create a new job with ERC20 token escrow
     */
    function createJobWithToken(
        address _provider,
        uint256 _amount
    ) external nonReentrant returns (uint256 jobId) {
        require(_provider != address(0), "Invalid provider");
        require(_amount > 0, "Amount must be > 0");

        require(
            paymentToken.transferFrom(msg.sender, address(this), _amount),
            "Transfer failed"
        );

        jobId = escrowJobCounter++;
        escrowJobs[jobId] = EscrowJob({
            requester: msg.sender,
            provider: _provider,
            amount: _amount,
            status: EscrowJobStatus.PENDING,
            proofHash: bytes32(0),
            createdAt: block.timestamp,
            completedAt: 0
        });

        emit EscrowJobCreated(jobId, msg.sender, _provider, _amount);
    }

    /**
     * @dev Create a new job with native ETH escrow
     */
    function createJobWithETH(
        address _provider
    ) external payable nonReentrant returns (uint256 jobId) {
        require(_provider != address(0), "Invalid provider");
        require(msg.value > 0, "Must send ETH");

        jobId = escrowJobCounter++;
        escrowJobs[jobId] = EscrowJob({
            requester: msg.sender,
            provider: _provider,
            amount: msg.value,
            status: EscrowJobStatus.PENDING,
            proofHash: bytes32(0),
            createdAt: block.timestamp,
            completedAt: 0
        });

        ethEscrow[jobId] = msg.value;

        emit EscrowJobCreated(jobId, msg.sender, _provider, msg.value);
        emit ETHDeposited(jobId, msg.value);
    }

    /**
     * @dev Complete job with ERC20 and release funds upon proof verification
     */
    function completeJobWithProof(
        uint256 _jobId,
        bytes32 _proofHash
    ) external nonReentrant {
        EscrowJob storage job = escrowJobs[_jobId];

        require(job.status == EscrowJobStatus.PENDING, "Job not pending");
        require(msg.sender == job.provider, "Only provider can complete");
        require(verifier.isProofVerified(_proofHash), "Proof not verified");

        uint256 fee = (job.amount * platformFeePercent) / 100;
        uint256 providerPayout = job.amount - fee;

        job.status = EscrowJobStatus.COMPLETED;
        job.proofHash = _proofHash;
        job.completedAt = block.timestamp;

        if (ethEscrow[_jobId] > 0) {
            // ETH payment
            uint256 ethFee = (ethEscrow[_jobId] * platformFeePercent) / 100;
            uint256 ethPayout = ethEscrow[_jobId] - ethFee;
            accumulatedFees += ethFee;
            ethEscrow[_jobId] = 0;
            
            (bool success, ) = payable(job.provider).call{value: ethPayout}("");
            require(success, "ETH transfer failed");
        } else {
            // ERC20 payment
            accumulatedFees += fee;
            require(
                paymentToken.transfer(job.provider, providerPayout),
                "Provider transfer failed"
            );
        }

        emit EscrowJobCompleted(_jobId, _proofHash, providerPayout);
    }

    /**
     * @dev Dispute a job (refunds requester)
     */
    function disputeJob(uint256 _jobId, string calldata _reason) external nonReentrant {
        EscrowJob storage job = escrowJobs[_jobId];

        require(job.status == EscrowJobStatus.PENDING, "Job not pending");
        require(
            msg.sender == job.requester || msg.sender == owner(),
            "Not authorized"
        );

        job.status = EscrowJobStatus.DISPUTED;

        if (ethEscrow[_jobId] > 0) {
            uint256 refundAmount = ethEscrow[_jobId];
            ethEscrow[_jobId] = 0;
            (bool success, ) = payable(job.requester).call{value: refundAmount}("");
            require(success, "ETH refund failed");
        } else {
            require(
                paymentToken.transfer(job.requester, job.amount),
                "Refund failed"
            );
        }

        emit EscrowJobDisputed(_jobId, _reason);
    }

    /**
     * @dev Withdraw accumulated platform fees
     */
    function withdrawFees() external onlyOwner {
        uint256 tokenFees = paymentToken.balanceOf(address(this));
        uint256 ethFees = accumulatedFees;
        
        if (tokenFees > 0) {
            require(paymentToken.transfer(feeRecipient, tokenFees), "Token withdrawal failed");
        }
        
        if (ethFees > 0) {
            accumulatedFees = 0;
            (bool success, ) = payable(feeRecipient).call{value: ethFees}("");
            require(success, "ETH withdrawal failed");
        }
        
        emit FeeWithdrawn(feeRecipient, tokenFees + ethFees);
    }

    /**
     * @dev Get job details
     */
    function getEscrowJob(uint256 _jobId) external view returns (EscrowJob memory) {
        return escrowJobs[_jobId];
    }

    /**
     * @dev Set platform fee (owner only, max 10%)
     */
    function setPlatformFee(uint256 _feePercent) external onlyOwner {
        require(_feePercent <= 10, "Fee too high");
        platformFeePercent = _feePercent;
    }

    /**
     * @dev Update verifier contract address
     */
    function setVerifier(address _newVerifier) external onlyOwner {
        verifier = IZKMLVerifier(_newVerifier);
    }

    /**
     * @dev Update fee recipient
     */
    function setFeeRecipient(address _newRecipient) external onlyOwner {
        require(_newRecipient != address(0), "Invalid recipient");
        feeRecipient = _newRecipient;
    }

    // Receive ETH
    receive() external payable {}
}

// ============================================================================
// MOCK ERC20 TOKEN (For Testing in Remix)
// ============================================================================

contract MockUSDC is IERC20 {
    string public name = "Mock USDC";
    string public symbol = "mUSDC";
    uint8 public decimals = 6;
    uint256 private _totalSupply;
    
    mapping(address => uint256) private _balances;
    mapping(address => mapping(address => uint256)) private _allowances;
    
    constructor() {
        // Mint 1 million tokens to deployer
        _mint(msg.sender, 1000000 * 10**decimals);
    }
    
    function totalSupply() external view override returns (uint256) {
        return _totalSupply;
    }
    
    function balanceOf(address account) external view override returns (uint256) {
        return _balances[account];
    }
    
    function transfer(address to, uint256 value) external override returns (bool) {
        require(_balances[msg.sender] >= value, "Insufficient balance");
        _balances[msg.sender] -= value;
        _balances[to] += value;
        emit Transfer(msg.sender, to, value);
        return true;
    }
    
    function allowance(address owner, address spender) external view override returns (uint256) {
        return _allowances[owner][spender];
    }
    
    function approve(address spender, uint256 value) external override returns (bool) {
        _allowances[msg.sender][spender] = value;
        emit Approval(msg.sender, spender, value);
        return true;
    }
    
    function transferFrom(address from, address to, uint256 value) external override returns (bool) {
        require(_balances[from] >= value, "Insufficient balance");
        require(_allowances[from][msg.sender] >= value, "Insufficient allowance");
        
        _balances[from] -= value;
        _balances[to] += value;
        _allowances[from][msg.sender] -= value;
        
        emit Transfer(from, to, value);
        return true;
    }
    
    function _mint(address account, uint256 value) internal {
        _totalSupply += value;
        _balances[account] += value;
        emit Transfer(address(0), account, value);
    }
    
    // Faucet for testing - mint free tokens
    function faucet(uint256 amount) external {
        _mint(msg.sender, amount);
    }
}

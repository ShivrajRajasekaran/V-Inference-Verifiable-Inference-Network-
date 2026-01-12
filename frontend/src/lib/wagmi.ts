"use client";

import { createConfig, http } from "wagmi";
import { type Chain } from "wagmi/chains";
import { injected } from "wagmi/connectors";

// Shardeum EVM Testnet Chain Definition
export const shardeumTestnet: Chain = {
    id: 8119,
    name: "Shardeum EVM Testnet",
    nativeCurrency: {
        decimals: 18,
        name: "Shardeum",
        symbol: "SHM",
    },
    rpcUrls: {
        default: { http: ["https://api-mezame.shardeum.org"] },
    },
    blockExplorers: {
        default: { name: "Shardeum Explorer", url: "https://explorer-mezame.shardeum.org" },
    },
    testnet: true,
};

// Wagmi configuration - Shardeum Only
export const config = createConfig({
    chains: [shardeumTestnet],
    connectors: [
        injected(), // MetaMask and other injected wallets
    ],
    transports: {
        [shardeumTestnet.id]: http("https://api-mezame.shardeum.org"),
    },
});

// ============================================
// SHARDEUM DEPLOYED CONTRACT ADDRESSES
// ============================================
export const SHARDEUM_CONTRACTS = {
    MockUSDC: "0x0117A0EcF95dE28CCc0486D45D5362e020434575",
    VInferenceAudit: "0xb3BD0a70eB7eAe91E6F23564d897C8098574e892",
} as const;

// Main contract address (VInferenceAudit for proof anchoring)
export const CONTRACT_ADDRESS = SHARDEUM_CONTRACTS.VInferenceAudit as `0x${string}`;

// ZKMLVerifier ABI
export const ZKML_VERIFIER_ABI = [
    {
        inputs: [
            { internalType: "bytes", name: "_proof", type: "bytes" },
            { internalType: "bytes", name: "_publicInputs", type: "bytes" },
            { internalType: "uint256", name: "_jobId", type: "uint256" },
        ],
        name: "verifyProof",
        outputs: [{ internalType: "bool", name: "isValid", type: "bool" }],
        stateMutability: "nonpayable",
        type: "function",
    },
    {
        inputs: [{ internalType: "bytes32", name: "_proofHash", type: "bytes32" }],
        name: "isProofVerified",
        outputs: [{ internalType: "bool", name: "", type: "bool" }],
        stateMutability: "view",
        type: "function",
    },
] as const;

// Contract ABI (audit functions)
export const CONTRACT_ABI = [
    {
        inputs: [
            { internalType: "bytes32", name: "proofHash", type: "bytes32" },
            { internalType: "string", name: "jobId", type: "string" },
        ],
        name: "anchorAudit",
        outputs: [{ internalType: "bool", name: "success", type: "bool" }],
        stateMutability: "nonpayable",
        type: "function",
    },
    {
        inputs: [
            { internalType: "string", name: "jobId", type: "string" },
            { internalType: "bytes32", name: "proofHash", type: "bytes32" },
        ],
        name: "verifyAudit",
        outputs: [
            { internalType: "bool", name: "valid", type: "bool" },
            { internalType: "bytes32", name: "onChainHash", type: "bytes32" },
        ],
        stateMutability: "nonpayable",
        type: "function",
    },
    {
        inputs: [{ internalType: "string", name: "jobId", type: "string" }],
        name: "auditExists",
        outputs: [{ internalType: "bool", name: "exists", type: "bool" }],
        stateMutability: "view",
        type: "function",
    },
    {
        inputs: [{ internalType: "string", name: "jobId", type: "string" }],
        name: "getAudit",
        outputs: [
            { internalType: "bytes32", name: "proofHash", type: "bytes32" },
            { internalType: "address", name: "auditor", type: "address" },
            { internalType: "uint256", name: "timestamp", type: "uint256" },
            { internalType: "uint256", name: "blockNumber", type: "uint256" },
            { internalType: "bool", name: "exists", type: "bool" },
        ],
        stateMutability: "view",
        type: "function",
    },
    {
        inputs: [],
        name: "totalAudits",
        outputs: [{ internalType: "uint256", name: "", type: "uint256" }],
        stateMutability: "view",
        type: "function",
    },
] as const;

// Chain info - Shardeum Only
export const SHARDEUM_CHAIN_ID = 8119;
export const SHARDEUM_EXPLORER = "https://explorer-mezame.shardeum.org";

// Helper to get explorer link
export function getExplorerTxLink(txHash: string): string {
    return `${SHARDEUM_EXPLORER}/tx/${txHash}`;
}

export function getExplorerAddressLink(address: string): string {
    return `${SHARDEUM_EXPLORER}/address/${address}`;
}

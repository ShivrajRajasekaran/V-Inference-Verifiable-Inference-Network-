# 🧠 V-OBLIVION: Decentralized AI Inference & ML Training Platform

<div align="center">

**AI Inference + ML Training Marketplace on Shardeum**

[![Shardeum](https://img.shields.io/badge/Shardeum-8119-00d4aa?style=for-the-badge)](https://shardeum.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=for-the-badge)](https://nextjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Python-009688?style=for-the-badge)](https://fastapi.tiangolo.com/)
[![ZKML](https://img.shields.io/badge/ZKML-Verified-6366f1?style=for-the-badge)]()

</div>

---

## 🌟 Overview

**V-OBLIVION** combines the best of V-Inference and OBLIVION to create a fully decentralized platform for:

- **🤖 AI Inference**: Run verified AI inference with ZKML proofs
- **🏋️ ML Training**: Submit training jobs processed by decentralized workers
- **🛒 Marketplace**: Trade inference access while keeping models private
- **⛓️ Blockchain**: All transactions anchored on Shardeum

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **ZKML Verification** | ZK-SNARK proofs for every inference |
| **Differential Privacy** | Mathematical privacy guarantees (ε=1.0) |
| **Decentralized Workers** | Python nodes process jobs trustlessly |
| **IPFS Storage** | Scripts, datasets, models on Pinata |
| **Shardeum Network** | Low-cost, high-speed EVM blockchain |
| **Staking System** | Workers stake collateral for honesty |

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              SHARDEUM EVM TESTNET (Chain ID: 8119)              │
│  • VOblivionManager.sol - Jobs, staking, rewards                │
│  • VInferenceAudit.sol - ZK proof anchoring                     │
└─────────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         │                                   │
         ▼                                   ▼
┌──────────────────────┐      ┌─────────────────────────────┐
│        IPFS          │      │     DECENTRALIZED WORKERS   │
│   (Pinata Gateway)   │      │     worker/                 │
│                      │      │                             │
│  • Training scripts  │      │  • Polls Shardeum chain     │
│  • Datasets          │      │  • Runs inference (ZKML)    │
│  • Trained models    │      │  • Trains models (PyTorch)  │
│  • ZK proofs         │      │  • Differential privacy     │
└──────────────────────┘      │  • Quality verification     │
                              └─────────────────────────────┘
                                         │
         ┌───────────────────────────────┘
         │
         ▼
┌─────────────────────────┐      ┌─────────────────────────┐
│    FASTAPI BACKEND      │      │    NEXT.JS FRONTEND     │
│    backend/             │      │    frontend/            │
│                         │      │                         │
│  • Job orchestration    │      │  • Dashboard            │
│  • ZKML verification    │      │  • Inference page       │
│  • API endpoints        │      │  • Training jobs        │
│  • Marketplace logic    │      │  • Worker management    │
└─────────────────────────┘      │  • Marketplace          │
                                 └─────────────────────────┘
```

## 📁 Project Structure

```
V-OBLIVION/
├── backend/                    # FastAPI Server
│   ├── app/
│   │   ├── api/               # REST endpoints
│   │   ├── core/              # Config, blockchain
│   │   └── services/          # ZKML, escrow
│   └── main.py
│
├── worker/                     # Decentralized Worker Node
│   ├── decentralized_worker.py # Main worker
│   ├── blockchain_client.py    # Shardeum client
│   ├── ipfs_client.py         # IPFS/Pinata
│   ├── privacy.py             # Differential privacy
│   ├── quality_verification.py # Quality checks
│   └── zk_proofs.py           # ZK proof generation
│
├── frontend/                   # Next.js 16 UI
│   └── src/app/
│       ├── dashboard/
│       │   ├── inference/     # AI inference
│       │   ├── marketplace/   # Buy/sell models
│       │   └── models/        # Model management
│       └── page.tsx
│
├── contracts/                  # Solidity contracts
│   └── VInference_Remix.sol
│
└── deploy/                     # Deployment scripts
    └── deploy_python.py
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.11+
- MetaMask with Shardeum SHM

### 1. Backend Setup

```bash
cd backend
pip install -r requirements.txt
python main.py
# API at http://localhost:8000
```

### 2. Frontend Setup

```bash
cd frontend
npm install
npm run dev
# UI at http://localhost:3000
```

### 3. Worker Setup (Optional)

```bash
cd worker
pip install -r requirements.txt
# Edit .env with your Shardeum wallet
python decentralized_worker.py
```

## 🔗 Contract Addresses (Shardeum)

| Contract | Address |
|----------|---------|
| VInferenceAudit | `0xb3BD0a70eB7eAe91E6F23564d897C8098574e892` |
| MockUSDC | `0x0117A0EcF95dE28CCc0486D45D5362e020434575` |

## 🦊 Add Shardeum to MetaMask

| Setting | Value |
|---------|-------|
| Network Name | Shardeum EVM Testnet |
| RPC URL | `https://api-mezame.shardeum.org` |
| Chain ID | `8119` |
| Symbol | `SHM` |
| Explorer | `https://explorer-mezame.shardeum.org` |

## 📖 How It Works

### AI Inference Flow
1. User submits inference request
2. Backend generates ZKML proof
3. Proof is anchored on Shardeum
4. User receives verified output

### ML Training Flow
1. Requester creates job with reward
2. Worker claims job (stakes 50%)
3. Worker downloads script/data from IPFS
4. Worker trains with differential privacy
5. Worker uploads model to IPFS
6. Worker submits proof on-chain
7. Smart contract pays worker

## 🛠️ Development

```bash
# Backend
cd backend && python main.py

# Frontend
cd frontend && npm run dev

# Worker
cd worker && python decentralized_worker.py
```

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines.

## 📄 License

MIT License

---

<div align="center">
Built with ❤️ by the V-OBLIVION Team
</div>

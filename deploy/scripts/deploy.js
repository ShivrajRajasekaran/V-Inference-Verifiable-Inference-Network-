const hre = require("hardhat");

async function main() {
    console.log("🚀 Deploying V-Inference Contracts to", hre.network.name);
    console.log("=".repeat(50));

    const [deployer] = await hre.ethers.getSigners();
    console.log("📝 Deploying with account:", deployer.address);

    const balance = await hre.ethers.provider.getBalance(deployer.address);
    console.log("💰 Account balance:", hre.ethers.formatEther(balance), "SHM/ETH");
    console.log("");

    // 1. Deploy MockUSDC (for testing)
    console.log("1️⃣  Deploying MockUSDC...");
    const MockUSDC = await hre.ethers.getContractFactory("MockUSDC");
    const mockUSDC = await MockUSDC.deploy();
    await mockUSDC.waitForDeployment();
    const mockUSDCAddress = await mockUSDC.getAddress();
    console.log("   ✅ MockUSDC deployed to:", mockUSDCAddress);

    // 2. Deploy ZKMLVerifier
    console.log("2️⃣  Deploying ZKMLVerifier...");
    const ZKMLVerifier = await hre.ethers.getContractFactory("ZKMLVerifier");
    const verifier = await ZKMLVerifier.deploy();
    await verifier.waitForDeployment();
    const verifierAddress = await verifier.getAddress();
    console.log("   ✅ ZKMLVerifier deployed to:", verifierAddress);

    // 3. Deploy NodeRegistry
    console.log("3️⃣  Deploying NodeRegistry...");
    const NodeRegistry = await hre.ethers.getContractFactory("NodeRegistry");
    const nodeRegistry = await NodeRegistry.deploy();
    await nodeRegistry.waitForDeployment();
    const nodeRegistryAddress = await nodeRegistry.getAddress();
    console.log("   ✅ NodeRegistry deployed to:", nodeRegistryAddress);

    // 4. Deploy InferenceMarketplace
    console.log("4️⃣  Deploying InferenceMarketplace...");
    const InferenceMarketplace = await hre.ethers.getContractFactory("InferenceMarketplace");
    const marketplace = await InferenceMarketplace.deploy();
    await marketplace.waitForDeployment();
    const marketplaceAddress = await marketplace.getAddress();
    console.log("   ✅ InferenceMarketplace deployed to:", marketplaceAddress);

    // 5. Deploy InferenceEscrow
    console.log("5️⃣  Deploying InferenceEscrow...");
    const feeRecipient = process.env.FEE_RECIPIENT || deployer.address;
    const InferenceEscrow = await hre.ethers.getContractFactory("InferenceEscrow");
    const escrow = await InferenceEscrow.deploy(mockUSDCAddress, verifierAddress, feeRecipient);
    await escrow.waitForDeployment();
    const escrowAddress = await escrow.getAddress();
    console.log("   ✅ InferenceEscrow deployed to:", escrowAddress);

    // Summary
    console.log("");
    console.log("=".repeat(50));
    console.log("🎉 DEPLOYMENT COMPLETE!");
    console.log("=".repeat(50));
    console.log("");
    console.log("📋 Contract Addresses:");
    console.log("─".repeat(50));
    console.log(`   MockUSDC:             ${mockUSDCAddress}`);
    console.log(`   ZKMLVerifier:         ${verifierAddress}`);
    console.log(`   NodeRegistry:         ${nodeRegistryAddress}`);
    console.log(`   InferenceMarketplace: ${marketplaceAddress}`);
    console.log(`   InferenceEscrow:      ${escrowAddress}`);
    console.log("─".repeat(50));
    console.log("");
    console.log("💡 Save these addresses for frontend integration!");

    // Return addresses for verification
    return {
        mockUSDC: mockUSDCAddress,
        verifier: verifierAddress,
        nodeRegistry: nodeRegistryAddress,
        marketplace: marketplaceAddress,
        escrow: escrowAddress,
    };
}

main()
    .then((addresses) => {
        console.log("\n✅ All contracts deployed successfully!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("❌ Deployment failed:", error);
        process.exit(1);
    });

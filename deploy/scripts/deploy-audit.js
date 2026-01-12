const hre = require("hardhat");

async function main() {
    console.log("🚀 Deploying VInferenceAudit to Shardeum EVM Testnet");
    console.log("=".repeat(50));

    try {
        const [deployer] = await hre.ethers.getSigners();
        console.log("📝 Deploying with account:", deployer.address);

        const balance = await hre.ethers.provider.getBalance(deployer.address);
        const balanceNumber = parseFloat(hre.ethers.formatEther(balance));
        console.log("💰 Account balance:", balanceNumber.toFixed(4), "SHM");
        console.log("");

        // Get network gas price to understand minimum required
        const feeData = await hre.ethers.provider.getFeeData();
        const networkGasPrice = feeData.gasPrice || BigInt(1000000000);
        console.log("⛓️ Network base fee:", hre.ethers.formatUnits(networkGasPrice, "gwei"), "gwei");

        // Use network price (required by Shardeum)
        const gasLimit = 800000n; // Optimized gas limit
        const estimatedCost = networkGasPrice * gasLimit;
        const estimatedCostEth = parseFloat(hre.ethers.formatEther(estimatedCost));

        console.log("⛽ Gas limit: 800,000");
        console.log("💵 Estimated cost:", estimatedCostEth.toFixed(4), "SHM");
        console.log("");

        if (balanceNumber < estimatedCostEth) {
            console.log("❌ Insufficient balance for deployment");
            console.log("   Need:", estimatedCostEth.toFixed(4), "SHM");
            console.log("   Have:", balanceNumber.toFixed(4), "SHM");
            console.log("");
            console.log("💡 Options:");
            console.log("   1. Get more SHM from faucet: https://faucet.shardeum.org/");
            console.log("   2. Wait for gas prices to normalize");
            console.log("   3. Use simulation mode instead");
            process.exit(1);
        }

        // Deploy VInferenceAudit
        console.log("📦 Deploying VInferenceAudit...");
        const VInferenceAudit = await hre.ethers.getContractFactory("VInferenceAudit");

        const audit = await VInferenceAudit.deploy({
            gasLimit: gasLimit,
            gasPrice: networkGasPrice
        });

        console.log("⏳ Waiting for deployment confirmation...");
        const txHash = audit.deploymentTransaction()?.hash;
        console.log("📄 Transaction hash:", txHash);

        await audit.waitForDeployment();
        const auditAddress = await audit.getAddress();

        console.log("");
        console.log("=".repeat(50));
        console.log("🎉 DEPLOYMENT COMPLETE!");
        console.log("=".repeat(50));
        console.log("");
        console.log("📋 VInferenceAudit Contract Address:");
        console.log("─".repeat(50));
        console.log(`   ${auditAddress}`);
        console.log("─".repeat(50));
        console.log("");
        console.log("💡 Copy this address to backend/app/core/config.py");

        return auditAddress;
    } catch (error) {
        console.error("❌ Error:", error.message);
        throw error;
    }
}

main()
    .then((address) => {
        console.log("\n✅ VInferenceAudit deployed successfully!");
        process.exit(0);
    })
    .catch((error) => {
        console.error("❌ Deployment failed:", error.message);
        process.exit(1);
    });

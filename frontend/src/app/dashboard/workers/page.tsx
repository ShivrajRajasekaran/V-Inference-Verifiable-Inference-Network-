"use client";

import { useState, useEffect } from "react";

// Icons
const ServerIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
    </svg>
);

const ShieldIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
    </svg>
);

const ChartIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
    </svg>
);

interface Worker {
    id: string;
    address: string;
    node_id: string;
    stake: number;
    completed_jobs: number;
    reputation: number;
    status: "active" | "idle" | "offline";
    last_seen: string;
}

export default function WorkersPage() {
    const [workers, setWorkers] = useState<Worker[]>([]);
    const [loading, setLoading] = useState(true);
    const [networkStats, setNetworkStats] = useState({
        total_workers: 0,
        active_workers: 0,
        total_stake: 0,
        total_completed: 0
    });

    useEffect(() => {
        fetchWorkers();
    }, []);

    const fetchWorkers = async () => {
        try {
            // Simulated workers for now
            const mockWorkers: Worker[] = [
                {
                    id: "1",
                    address: "0xF19D787BE014d43c04FeE3862485C47E792c92F3",
                    node_id: "WORKER-A1B2C3D4",
                    stake: 10.5,
                    completed_jobs: 127,
                    reputation: 98,
                    status: "active",
                    last_seen: new Date().toISOString()
                },
                {
                    id: "2",
                    address: "0x1234567890abcdef1234567890abcdef12345678",
                    node_id: "WORKER-E5F6G7H8",
                    stake: 5.2,
                    completed_jobs: 84,
                    reputation: 95,
                    status: "idle",
                    last_seen: new Date(Date.now() - 60000).toISOString()
                },
                {
                    id: "3",
                    address: "0xabcdef1234567890abcdef1234567890abcdef12",
                    node_id: "WORKER-I9J0K1L2",
                    stake: 8.0,
                    completed_jobs: 56,
                    reputation: 92,
                    status: "active",
                    last_seen: new Date().toISOString()
                }
            ];

            setWorkers(mockWorkers);
            setNetworkStats({
                total_workers: mockWorkers.length,
                active_workers: mockWorkers.filter(w => w.status === "active").length,
                total_stake: mockWorkers.reduce((sum, w) => sum + w.stake, 0),
                total_completed: mockWorkers.reduce((sum, w) => sum + w.completed_jobs, 0)
            });
        } catch (error) {
            console.error("Error fetching workers:", error);
        } finally {
            setLoading(false);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "active": return "bg-green-500";
            case "idle": return "bg-yellow-500";
            default: return "bg-gray-500";
        }
    };

    return (
        <div className="space-y-8 animate-fade-in">
            {/* Header */}
            <div>
                <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                    <ServerIcon /> Worker Network
                </h1>
                <p className="text-[var(--foreground-muted)]">
                    Decentralized worker nodes processing ML jobs on Shardeum
                </p>
            </div>

            {/* Network Stats */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-[var(--primary-500)]/20">
                            <ServerIcon />
                        </div>
                        <span className="text-[var(--foreground-muted)]">Total Workers</span>
                    </div>
                    <div className="text-3xl font-bold">{networkStats.total_workers}</div>
                </div>

                <div className="glass-card p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-green-500/20">
                            <div className="w-3 h-3 rounded-full bg-green-500 animate-pulse"></div>
                        </div>
                        <span className="text-[var(--foreground-muted)]">Active Now</span>
                    </div>
                    <div className="text-3xl font-bold text-green-400">{networkStats.active_workers}</div>
                </div>

                <div className="glass-card p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-[var(--secondary-500)]/20">
                            <ShieldIcon />
                        </div>
                        <span className="text-[var(--foreground-muted)]">Total Staked</span>
                    </div>
                    <div className="text-3xl font-bold">{networkStats.total_stake.toFixed(2)} SHM</div>
                </div>

                <div className="glass-card p-5">
                    <div className="flex items-center gap-3 mb-2">
                        <div className="p-2 rounded-lg bg-[var(--accent-500)]/20">
                            <ChartIcon />
                        </div>
                        <span className="text-[var(--foreground-muted)]">Jobs Completed</span>
                    </div>
                    <div className="text-3xl font-bold">{networkStats.total_completed}</div>
                </div>
            </div>

            {/* Workers Table */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">Active Workers</h2>

                {loading ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[var(--primary-500)]"></div>
                    </div>
                ) : workers.length === 0 ? (
                    <div className="text-center py-12 text-[var(--foreground-muted)]">
                        <ServerIcon />
                        <p className="mt-2">No workers registered yet</p>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="text-left text-sm text-[var(--foreground-muted)] border-b border-[var(--glass-border)]">
                                    <th className="pb-3 font-medium">Status</th>
                                    <th className="pb-3 font-medium">Node ID</th>
                                    <th className="pb-3 font-medium">Address</th>
                                    <th className="pb-3 font-medium">Stake</th>
                                    <th className="pb-3 font-medium">Completed</th>
                                    <th className="pb-3 font-medium">Reputation</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {workers.map((worker) => (
                                    <tr
                                        key={worker.id}
                                        className="border-b border-[var(--glass-border)]/50 hover:bg-[var(--glass-bg)] transition-colors"
                                    >
                                        <td className="py-4">
                                            <div className="flex items-center gap-2">
                                                <div className={`w-2 h-2 rounded-full ${getStatusColor(worker.status)}`}></div>
                                                <span className="capitalize">{worker.status}</span>
                                            </div>
                                        </td>
                                        <td className="py-4 font-mono text-[var(--primary-400)]">
                                            {worker.node_id}
                                        </td>
                                        <td className="py-4 font-mono text-xs">
                                            {worker.address.slice(0, 10)}...{worker.address.slice(-8)}
                                        </td>
                                        <td className="py-4">{worker.stake.toFixed(2)} SHM</td>
                                        <td className="py-4">{worker.completed_jobs}</td>
                                        <td className="py-4">
                                            <div className="flex items-center gap-2">
                                                <div className="w-16 h-2 bg-[var(--glass-bg)] rounded-full overflow-hidden">
                                                    <div
                                                        className="h-full bg-[var(--secondary-500)] rounded-full"
                                                        style={{ width: `${worker.reputation}%` }}
                                                    ></div>
                                                </div>
                                                <span>{worker.reputation}%</span>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* Run a Worker Section */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">🖥️ Run Your Own Worker</h2>
                <div className="space-y-4">
                    <p className="text-[var(--foreground-muted)]">
                        Earn SHM by processing ML jobs with your compute power!
                    </p>

                    <div className="p-4 rounded-xl bg-[var(--dark-900)] font-mono text-sm">
                        <div className="text-[var(--foreground-muted)] mb-2"># Clone and setup</div>
                        <div className="text-green-400">$ cd worker</div>
                        <div className="text-green-400">$ pip install -r requirements.txt</div>
                        <div className="text-green-400">$ cp .env.example .env</div>
                        <div className="text-[var(--foreground-muted)] mt-2"># Edit .env with your private key</div>
                        <div className="text-green-400">$ python decentralized_worker.py</div>
                    </div>

                    <div className="grid md:grid-cols-3 gap-4 mt-4">
                        <div className="p-4 rounded-xl bg-[var(--glass-bg)]">
                            <div className="text-lg font-semibold mb-1">💰 Earn Rewards</div>
                            <div className="text-sm text-[var(--foreground-muted)]">
                                Get paid in SHM for each completed job
                            </div>
                        </div>
                        <div className="p-4 rounded-xl bg-[var(--glass-bg)]">
                            <div className="text-lg font-semibold mb-1">🔒 Stake to Participate</div>
                            <div className="text-sm text-[var(--foreground-muted)]">
                                Minimum 0.001 SHM stake required
                            </div>
                        </div>
                        <div className="p-4 rounded-xl bg-[var(--glass-bg)]">
                            <div className="text-lg font-semibold mb-1">🛡️ Privacy Enabled</div>
                            <div className="text-sm text-[var(--foreground-muted)]">
                                Differential privacy (ε=1.0) by default
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

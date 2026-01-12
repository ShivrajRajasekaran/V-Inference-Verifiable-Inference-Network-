"use client";

import { useState, useEffect } from "react";

// Icons
const BrainIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
    </svg>
);

const UploadIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
    </svg>
);

const ClockIcon = () => (
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
);

const CubeIcon = () => (
    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
    </svg>
);

interface TrainingJob {
    id: string;
    status: "pending" | "processing" | "completed" | "failed";
    script_url: string;
    dataset_url: string;
    reward: number;
    created_at: string;
    worker_address?: string;
    result_url?: string;
}

export default function TrainingPage() {
    const [jobs, setJobs] = useState<TrainingJob[]>([]);
    const [loading, setLoading] = useState(true);
    const [showCreateModal, setShowCreateModal] = useState(false);

    // Form state
    const [scriptUrl, setScriptUrl] = useState("");
    const [datasetUrl, setDatasetUrl] = useState("");
    const [reward, setReward] = useState("0.01");
    const [creating, setCreating] = useState(false);

    useEffect(() => {
        fetchJobs();
    }, []);

    const fetchJobs = async () => {
        try {
            // Simulated jobs for now - will connect to backend
            const mockJobs: TrainingJob[] = [
                {
                    id: "job-001",
                    status: "completed",
                    script_url: "ipfs://QmTrainScript...",
                    dataset_url: "ipfs://QmDataset...",
                    reward: 0.05,
                    created_at: new Date().toISOString(),
                    worker_address: "0x1234...5678",
                    result_url: "ipfs://QmResult..."
                },
                {
                    id: "job-002",
                    status: "processing",
                    script_url: "ipfs://QmTrainScript2...",
                    dataset_url: "ipfs://QmDataset2...",
                    reward: 0.03,
                    created_at: new Date().toISOString(),
                    worker_address: "0xabcd...efgh"
                },
                {
                    id: "job-003",
                    status: "pending",
                    script_url: "ipfs://QmTrainScript3...",
                    dataset_url: "ipfs://QmDataset3...",
                    reward: 0.02,
                    created_at: new Date().toISOString()
                }
            ];
            setJobs(mockJobs);
        } catch (error) {
            console.error("Error fetching jobs:", error);
        } finally {
            setLoading(false);
        }
    };

    const handleCreateJob = async () => {
        if (!scriptUrl || !datasetUrl || !reward) return;

        setCreating(true);
        try {
            // TODO: Call backend API to create job on Shardeum
            console.log("Creating job:", { scriptUrl, datasetUrl, reward });

            // Simulate creation
            await new Promise(resolve => setTimeout(resolve, 2000));

            const newJob: TrainingJob = {
                id: `job-${Date.now()}`,
                status: "pending",
                script_url: scriptUrl,
                dataset_url: datasetUrl,
                reward: parseFloat(reward),
                created_at: new Date().toISOString()
            };

            setJobs([newJob, ...jobs]);
            setShowCreateModal(false);
            setScriptUrl("");
            setDatasetUrl("");
            setReward("0.01");
        } catch (error) {
            console.error("Error creating job:", error);
        } finally {
            setCreating(false);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case "completed": return "badge-success";
            case "processing": return "badge-primary";
            case "failed": return "bg-red-500/20 text-red-400";
            default: return "bg-yellow-500/20 text-yellow-400";
        }
    };

    return (
        <div className="space-y-8 animate-fade-in">
            {/* Header */}
            <div className="flex items-center justify-between">
                <div>
                    <h1 className="text-3xl font-bold mb-2 flex items-center gap-3">
                        <BrainIcon /> ML Training Jobs
                    </h1>
                    <p className="text-[var(--foreground-muted)]">
                        Submit ML training jobs to decentralized workers on Shardeum
                    </p>
                </div>
                <button
                    onClick={() => setShowCreateModal(true)}
                    className="btn btn-primary"
                >
                    <UploadIcon /> Create Job
                </button>
            </div>

            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div className="glass-card p-4 text-center">
                    <div className="text-3xl font-bold text-[var(--primary-400)]">{jobs.length}</div>
                    <div className="text-sm text-[var(--foreground-muted)]">Total Jobs</div>
                </div>
                <div className="glass-card p-4 text-center">
                    <div className="text-3xl font-bold text-yellow-400">
                        {jobs.filter(j => j.status === "pending").length}
                    </div>
                    <div className="text-sm text-[var(--foreground-muted)]">Pending</div>
                </div>
                <div className="glass-card p-4 text-center">
                    <div className="text-3xl font-bold text-blue-400">
                        {jobs.filter(j => j.status === "processing").length}
                    </div>
                    <div className="text-sm text-[var(--foreground-muted)]">Processing</div>
                </div>
                <div className="glass-card p-4 text-center">
                    <div className="text-3xl font-bold text-[var(--secondary-400)]">
                        {jobs.filter(j => j.status === "completed").length}
                    </div>
                    <div className="text-sm text-[var(--foreground-muted)]">Completed</div>
                </div>
            </div>

            {/* Jobs Table */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">Training Jobs</h2>

                {loading ? (
                    <div className="flex items-center justify-center h-32">
                        <div className="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-[var(--primary-500)]"></div>
                    </div>
                ) : jobs.length === 0 ? (
                    <div className="text-center py-12 text-[var(--foreground-muted)]">
                        <CubeIcon />
                        <p className="mt-2">No training jobs yet</p>
                        <button
                            onClick={() => setShowCreateModal(true)}
                            className="mt-4 text-[var(--primary-400)] hover:underline"
                        >
                            Create your first job
                        </button>
                    </div>
                ) : (
                    <div className="overflow-x-auto">
                        <table className="w-full">
                            <thead>
                                <tr className="text-left text-sm text-[var(--foreground-muted)] border-b border-[var(--glass-border)]">
                                    <th className="pb-3 font-medium">Job ID</th>
                                    <th className="pb-3 font-medium">Status</th>
                                    <th className="pb-3 font-medium">Reward</th>
                                    <th className="pb-3 font-medium">Worker</th>
                                    <th className="pb-3 font-medium">Created</th>
                                </tr>
                            </thead>
                            <tbody className="text-sm">
                                {jobs.map((job) => (
                                    <tr
                                        key={job.id}
                                        className="border-b border-[var(--glass-border)]/50 hover:bg-[var(--glass-bg)] transition-colors"
                                    >
                                        <td className="py-3 font-mono">{job.id}</td>
                                        <td className="py-3">
                                            <span className={`badge ${getStatusColor(job.status)}`}>
                                                {job.status}
                                            </span>
                                        </td>
                                        <td className="py-3">{job.reward} SHM</td>
                                        <td className="py-3 font-mono text-xs">
                                            {job.worker_address || "-"}
                                        </td>
                                        <td className="py-3 flex items-center gap-1 text-[var(--foreground-muted)]">
                                            <ClockIcon />
                                            {new Date(job.created_at).toLocaleString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>

            {/* How It Works */}
            <div className="glass-card p-6">
                <h2 className="text-xl font-semibold mb-4">How It Works</h2>
                <div className="grid md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-xl bg-[var(--glass-bg)] text-center">
                        <div className="text-2xl mb-2">1️⃣</div>
                        <div className="font-medium">Upload Script & Data</div>
                        <div className="text-sm text-[var(--foreground-muted)]">
                            Push to IPFS via Pinata
                        </div>
                    </div>
                    <div className="p-4 rounded-xl bg-[var(--glass-bg)] text-center">
                        <div className="text-2xl mb-2">2️⃣</div>
                        <div className="font-medium">Create Job</div>
                        <div className="text-sm text-[var(--foreground-muted)]">
                            Submit on-chain with reward
                        </div>
                    </div>
                    <div className="p-4 rounded-xl bg-[var(--glass-bg)] text-center">
                        <div className="text-2xl mb-2">3️⃣</div>
                        <div className="font-medium">Worker Claims</div>
                        <div className="text-sm text-[var(--foreground-muted)]">
                            Stakes & trains with DP
                        </div>
                    </div>
                    <div className="p-4 rounded-xl bg-[var(--glass-bg)] text-center">
                        <div className="text-2xl mb-2">4️⃣</div>
                        <div className="font-medium">Get Result</div>
                        <div className="text-sm text-[var(--foreground-muted)]">
                            Model on IPFS, verified
                        </div>
                    </div>
                </div>
            </div>

            {/* Create Job Modal */}
            {showCreateModal && (
                <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
                    <div
                        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                        onClick={() => setShowCreateModal(false)}
                    />
                    <div className="glass-card w-full max-w-md p-6 relative z-10 animate-fade-in">
                        <h2 className="text-xl font-bold mb-6 flex items-center gap-2">
                            <BrainIcon /> Create Training Job
                        </h2>

                        <div className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium mb-2">
                                    Training Script (IPFS URL)
                                </label>
                                <input
                                    type="text"
                                    value={scriptUrl}
                                    onChange={(e) => setScriptUrl(e.target.value)}
                                    placeholder="ipfs://Qm... or https://..."
                                    className="input"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-2">
                                    Dataset (IPFS URL)
                                </label>
                                <input
                                    type="text"
                                    value={datasetUrl}
                                    onChange={(e) => setDatasetUrl(e.target.value)}
                                    placeholder="ipfs://Qm... or https://..."
                                    className="input"
                                />
                            </div>

                            <div>
                                <label className="block text-sm font-medium mb-2">
                                    Reward (SHM)
                                </label>
                                <input
                                    type="number"
                                    value={reward}
                                    onChange={(e) => setReward(e.target.value)}
                                    min="0.001"
                                    step="0.001"
                                    className="input"
                                />
                            </div>

                            <div className="p-4 rounded-xl bg-[var(--glass-bg)] text-sm">
                                <div className="flex justify-between mb-1">
                                    <span>Reward</span>
                                    <span>{reward} SHM</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Worker Stake (50%)</span>
                                    <span>{(parseFloat(reward) * 0.5).toFixed(4)} SHM</span>
                                </div>
                            </div>

                            <div className="flex gap-3">
                                <button
                                    onClick={() => setShowCreateModal(false)}
                                    className="btn flex-1"
                                    style={{ background: "var(--glass-bg)" }}
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={handleCreateJob}
                                    disabled={!scriptUrl || !datasetUrl || creating}
                                    className="btn btn-primary flex-1 disabled:opacity-50"
                                >
                                    {creating ? (
                                        <div className="animate-spin rounded-full h-5 w-5 border-t-2 border-b-2 border-white"></div>
                                    ) : (
                                        "Create Job"
                                    )}
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

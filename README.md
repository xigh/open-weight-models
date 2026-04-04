# Open Weight Models

A curated list of open-weight AI models with commercially exploitable licenses, verified benchmarks, and no geographic restrictions.

**Selection criteria:**

1. **Commercially exploitable license**, no geographic restriction (EU ok)
2. **Total size < 200B** parameters
3. **Released after April 2024**

This excludes Llama 4 (EU exclusion), Qwen 3.6 Plus (closed-source), full DeepSeek V3/R1 (671B), and others. See [Rejected models](#rejected-models) for details.

> Maintained by [Philippe Anel](https://philippe-anel.fr). Last updated: April 2026.

---

## Table of Contents

- [LLMs](#llms)
  - [Generalists](#generalists)
  - [Code](#code)
  - [Reasoning](#reasoning)
  - [Compact / Edge](#compact--edge)
  - [Long context](#long-context)
  - [Alternative architectures](#alternative-architectures)
- [Specialized](#specialized)
  - [Theorem provers (Lean 4)](#theorem-provers-lean-4)
  - [GUI agents](#gui-agents)
  - [Search agents](#search-agents)
  - [Tool calling](#tool-calling)
  - [Rust](#rust)
- [Observations](#observations)
- [Benchmarks reference](#benchmarks-reference)
- [Licenses](#licenses)
- [How to choose](#how-to-choose)
- [Rejected models](#rejected-models)
- [Contributing](#contributing)

---

## LLMs

### Generalists

| Model | Publisher | Active | Total | Arch | Ctx | License | Key scores |
|-------|-----------|--------|-------|------|-----|---------|-----------|
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | Google | 31B | 31B | Dense | 256K | Apache 2.0 | GPQA 84.3, MMLU-Pro 85.2 |
| [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | Alibaba | 27B | 27B | Dense | 128K | Apache 2.0 | 201 languages |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | Alibaba | 9B | 9B | Dense | 128K | Apache 2.0 | GPQA 81.7 (9B!) |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) | Alibaba | 10B | 122B | MoE | 256K | Apache 2.0 | 201 languages, multimodal |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | OpenAI | 5.1B | 117B | MoE | 128K | Apache 2.0 | GPQA 80.9, Codeforces 2622, AIME 96.6% |
| [GPT-OSS-20B](https://huggingface.co/openai/GPT-OSS-20B) | OpenAI | 3.6B | 21B | MoE | 128K | Apache 2.0 | AIME 96%, fits 16GB |
| [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-128K) | Mistral | 6B | 119B | MoE | 256K | Apache 2.0 | GPQA 71.2, unified instruct/reasoning/coding |
| [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5) | Zhipu AI | 12B | 106B | MoE | 128K | MIT | MATH-500 98.1%, MMLU-Pro 81.4 |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | Alibaba | 32B | 32B | Dense | 128K | Apache 2.0 | AIME ~80%, reasoning RL |
| [DeepSeek R1-Distill-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek | 32B | 32B | Dense | 128K | MIT | Beats o1-mini |
| [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | StepFun | 11B | 196B | MoE | 262K | Apache 2.0 | SWE-bench 74.4%, 350 tok/s |

### Code

| Model | SWE-bench | Codeforces | Active | License |
|-------|-----------|-----------|--------|---------|
| [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | **74.4%** | -- | 11B | Apache 2.0 |
| [Devstral 2](https://huggingface.co/mistralai/Devstral-2) | 72.2% | -- | ~12B | MIT modified |
| [Qwen3-Coder-Next 80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) | 70.6% | -- | 3B | Apache 2.0 |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 69.6% | -- | 32B | Apache 2.0 |
| [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-2505) | 68.0% | -- | 24B | Apache 2.0 |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 62.4% | **2622** | 5.1B | Apache 2.0 |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | -- | 2150 | 31B | Apache 2.0 |

> SWE-bench = real bugs in real codebases. Codeforces = algorithmic competition. Different skills.

### Reasoning

#### GPQA Diamond (doctoral-level, 198 questions)

| Model | GPQA | Active |
|-------|------|--------|
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | 84.3 | 31B |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | 82.3 | 3.8B |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 81.7 | 9B |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 80.9 | 5.1B |
| [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5) | 75.0 | 12B |
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 73.0 | 3.5B |
| [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-128K) | 71.2 | 6B |

#### Math (AIME)

| Model | AIME | Conditions | Active |
|-------|------|-----------|--------|
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | **99.2%** | 2025, with tools | 3.5B |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 96.6% | 2024, with tools | 5.1B |
| [GPT-OSS-20B](https://huggingface.co/openai/GPT-OSS-20B) | 96.0% | 2024, with tools | 3.6B |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | 89.2% | 2026 | 31B |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | 88.3% | 2026 | 3.8B |
| [Ministral 14B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 85.0% | 2025 | 14B |
| [Nemotron Nano 9B v2](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) | 97.8% | MATH-500, /think mode | 9B |

> AIME versions (2024/2025/2026) are not comparable. Each year is harder.

### Compact / Edge

Models that run on smartphones, laptops, or edge devices.

| Model | Active | VRAM Q4 | Strength | License |
|-------|--------|---------|----------|---------|
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B | ~4 GB | Multimodal + audio | Apache 2.0 |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | 4.5B | ~6 GB | Multimodal + audio | Apache 2.0 |
| [Phi-4-mini](https://huggingface.co/microsoft/phi-4) | 3.8B | ~2 GB | MATH-500 92.5% | MIT |
| [Phi-4-multimodal](https://huggingface.co/microsoft/phi-4) | 5.6B | ~3 GB | Text + image + audio | MIT |
| [Ministral 3B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 3B | ~2 GB | Vision + reasoning, 256K ctx | Apache 2.0 |
| [Ministral 8B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 8B | ~5 GB | AIME 78.7%, vision | Apache 2.0 |
| [Ministral 14B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 14B | ~8 GB | AIME 85%, vision, 256K ctx | Apache 2.0 |

### Long context

| Model | Max ctx | RULER 1M | Architecture | Active | License |
|-------|---------|---------|-------------|--------|---------|
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1M | **86.3%** | Mamba/MoE | 3.5B | Nemotron OML |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 1M | -- | Mamba/MoE | 12B | Nemotron OML |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | 256K | -- | SSM+Transformer/MoE | 12B | Jamba OML |

> Many models claim "1M context" without publishing RULER scores at that length. Without measurement, it's marketing.

### Alternative architectures

Non-Transformer or hybrid models.

| Model | Architecture | Active | Key metric | License |
|-------|-------------|--------|-----------|---------|
| [Granite 4.0](https://huggingface.co/ibm-granite/granite-4.0-h-small) | 90% Mamba-2 / 10% Attention | 3-9B | 70% memory reduction, 2x speed | Apache 2.0 |
| [LFM2](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | Convolutions + grouped attention | 2.3B | 112 tok/s CPU, 2x Qwen3 | LFM Open v1.0 |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | Mamba + Transformer + MoE | 12B | 2.5x Transformer speed | Jamba OML |

---

## Specialized

### Theorem provers (Lean 4)

| Model | miniF2F | PutnamBench | Active | License |
|-------|---------|-----------|--------|---------|
| [BFS-Prover-V2-32B](https://huggingface.co/ByteDance-Seed/BFS-Prover-V2) | **95.0%** | -- | 32B | Apache 2.0 |
| [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) | 90.4% | **#1** | 32B | Apache 2.0 |
| [DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B) | 88.9% | -- | 7B | MIT |
| [Leanstral](https://huggingface.co/mistralai/Leanstral) | -- | -- | 32B | Apache 2.0 |
| [Kimina-Prover-72B](https://huggingface.co/MoonshotAI/Kimina-Prover-Preview-Distill-72B) | 84.0% | -- | 72B | MIT |
| [Leanabell-Prover-V2-7B](https://huggingface.co/leanbell/Leanabell-Prover-V2-7B) | 78.2% | -- | 7B | Apache 2.0 |

> Lean 4 proofs are verified by the compiler. Either correct or rejected. Zero hallucination on mathematical correctness.

> The sweet spot is 32B: BFS-Prover (95%) and Goedel-V2 (90.4%) both beat the 72B Kimina (84%).

### GUI agents

| Model | ScreenSpot | OSWorld | Active | License |
|-------|-----------|---------|--------|---------|
| [UI-TARS-1.5-7B](https://huggingface.co/bytedance-research/UI-TARS-1.5-7B) | **94.2%** | 42.5 | 7B | Apache 2.0 |
| [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | 84.7 | -- | 7B | Apache 2.0 |
| [ShowUI-2B](https://huggingface.co/showlab/ShowUI-2B) | -- | -- | 2B | MIT |

> UI-TARS-7B beats Claude (87.6%) on ScreenSpot. 7B, Apache 2.0, runs on a laptop.

### Search agents

| Model | Specialty | Active | License |
|-------|----------|--------|---------|
| [WebThinker-32B](https://huggingface.co/RUC-NLPIR/WebThinker-32B) | RL web search, beats Gemini Deep Research | 32B | Apache 2.0 |
| [DeepResearcher-7B](https://huggingface.co/GAIR/DeepResearcher-7B) | Emergent multi-step planning via RL | 7B | Apache 2.0 |
| [Search-R1](https://github.com/PeterGriffinJin/Search-R1) | Framework: teach any LLM to search (+26% on 7B) | any | Apache 2.0 |

### Tool calling

| Model | BFCL | Active | License |
|-------|------|--------|---------|
| [Hammer2.1-7B](https://huggingface.co/MadeAgents/Hammer2.1-7b) | #1 | 7B | CC-BY-NC 4.0 |
| [xLAM-8B](https://huggingface.co/Salesforce/xLAM-8B) | #1 (alternate) | 8B | CC-BY-NC 4.0 |
| [Hammer-0.5B](https://huggingface.co/MadeAgents/Hammer2.1-0.5b) | On-device | 0.5B | CC-BY-NC 4.0 |

> Specialized tool-calling models clearly beat generalists. xLAM-8B beats GPT-4o on BFCL.

### Rust

| Model | Strandset-Rust | RustEvo2 | Active | License |
|-------|---------------|---------|--------|---------|
| [Strand-Rust-Coder-14B](https://huggingface.co/strand-ai/Strand-Rust-Coder-14B) | **0.50** | **0.43** | 14B | Apache 2.0 (base) |

> Beats GPT-5-Codex and Claude Sonnet 4.5 on Rust benchmarks. Fine-tuned on 191K examples from 2,383 crates.

---

## Observations

Patterns observed across 60+ models. Not definitive truths.

### Architecture

- **Dense dies above 35B.** Every model > 35B released in 2025-2026 is MoE. The quality/compute ratio of MoE has made dense obsolete at this scale.

- **Parameter count is no longer the determining factor.** Qwen3.5-9B (9B) beats GPT-OSS-120B (5.1B active, 117B total) on GPQA Diamond.

- **The 40-79B segment is thinning out.** New models tend to jump from ~35B to ~120B total via MoE. The segment still has strong entries (Qwen 2.5-72B, R1-Distill-70B, Jamba 1.6 Mini 52B), but no new entrants in 2025-2026.

- **Qwen is the de facto base model** for fine-tuning. BFS-Prover, Goedel-Prover, Kimina-Prover, most community distillations: all built on Qwen. The ResNet of LLMs.

### Benchmarks

- **GPQA Diamond is the most discriminating benchmark** for reasoning: 198 doctoral-level questions, impossible to solve by retrieval.

- **SWE-bench vs Codeforces measure different things.** GPT-OSS-120B dominates competition (ELO 2622) but gets beaten on real bugs by Step-3.5-Flash (74.4% vs 62.4%).

- **Many models claim "1M context" without RULER scores** at that length. Without measurement, it's marketing.

- **AIME versions (2024/2025/2026) are not comparable.** Each year is harder. Only compare within the same version.

### Specialization

- **Specialized models dominate on narrow tasks.** UI-TARS-7B beats Claude on GUI (94.2% vs 87.6%). BFS-Prover-32B beats DeepSeek-671B on theorem proving (95% vs 88.9%).

- **The sweet spot for theorem proving is 32B.** Method (tree search, self-correction) compensates for size.

- **Domain-specific models (medical, legal, finance) are less mature** than code/math specialists. Generalists often outperform them on domain benchmarks. Specialization helps mainly for specific vocabulary, regulatory compliance, and private data fine-tuning.

### Licenses

- **Gemma 4 under Apache 2.0 is a turning point.** Google moved from a restrictive custom license to standard open-source for the first time.

- **Llama 4 excludes the EU** for multimodal models. The Llama-Nemotron-Super-49B inherits this exclusion despite NVIDIA's own license.

- **"Open-weight" is more nuanced than "open-source".** Llama 4 is technically open-weight but unexploitable in EU. Always check the fine print.

---

## Benchmarks reference

What each benchmark measures and when to trust it.

| Benchmark | Measures | Questions | Trust level |
|-----------|---------|-----------|-------------|
| GPQA Diamond | Doctoral reasoning | 198 | High (hard to game) |
| MMLU-Pro | Broad knowledge | 12K | Medium (10 choices, harder than MMLU) |
| AIME | Math competition | 30/year | High (but version-dependent) |
| MATH-500 | Math reasoning | 500 | Medium |
| SWE-bench Verified | Real bug fixing | 500 | High (real codebases) |
| Codeforces | Algorithmic competition | Varies | High (ELO system) |
| LiveCodeBench | Fresh coding problems | Rotating | High (not in training data) |
| RULER | Long context retrieval | Varies | High (parametric by length) |
| BFCL | Function/tool calling | 2K+ | High |
| miniF2F | Theorem proving (Lean 4) | 488 | Very high (compiler-verified) |
| ScreenSpot | GUI element grounding | 1.2K | High |

---

## Licenses

| License | Models | Commercial | EU | Patent grant | OSI |
|---------|--------|-----------|-----|-------------|-----|
| Apache 2.0 | Gemma 4, Qwen 3/3.5, GPT-OSS, Ministral, Step-3.5-Flash | Yes | Yes | Yes | Yes |
| MIT | GLM-4.5-Air, DeepSeek R1-Distill, Phi-4 | Yes | Yes | No (implicit) | Yes |
| Nemotron OML | Nemotron 3 Nano/Super | Yes | Yes | Yes | No |
| Jamba OML | Jamba 1.6 | Yes | Yes | -- | No |
| LFM Open v1.0 | LFM2 | Yes (< $10M) | Yes | -- | No |

---

## How to choose

| Constraint | Recommendation |
|-----------|---------------|
| Smartphone / edge (< 4 GB) | Gemma 4 E2B, Phi-4-mini, Ministral 3B |
| Laptop 16 GB | GPT-OSS-20B, Ministral 14B, Gemma 4 26B-A4B |
| Desktop 24 GB | Gemma 4 31B, DeepSeek R1-Distill-32B, Devstral Small 2 |
| Server single-GPU (80 GB) | GPT-OSS-120B |
| Server multi-GPU | Step-3.5-Flash, Nemotron 3 Super, Qwen3.5-122B |
| Long context (> 256K) | Nemotron 3 Nano (1M, RULER 86.3%) |
| Math | Nemotron Nano 9B v2 (/think mode), GPT-OSS-120B |
| Code (real bugs) | Step-3.5-Flash, Devstral Small 2 |
| Code (competition) | GPT-OSS-120B (Codeforces 2622) |
| Multilingual (100+ langs) | Qwen 3.5 (201), Qwen 3 (119) |
| Theorem proving | BFS-Prover-V2-32B (95% miniF2F) |
| GUI automation | UI-TARS-1.5-7B (94.2% ScreenSpot) |
| Throughput | Step-3.5-Flash (350 tok/s) |

---

## Rejected models

| Model | License | Reason |
|-------|---------|--------|
| Llama 4 (Maverick, Scout) | Llama Community License | EU exclusion for multimodal models |
| Llama-Nemotron-Super-49B | Llama 3.3 License | Inherits EU exclusion |
| Qwen 3.6 Plus | Proprietary | Closed-source, API-only |
| Codestral | Non-commercial | Research only |
| Falcon 3 | Ambiguous | Potential 10% royalty |
| Kimi K2.5 | Modified MIT (100M MAU) | User threshold |
| MiniMax M2.5 | Modified MIT | Custom restrictions |
| DeepSeek V3/R1 (full) | MIT | > 200B total (671B) |
| Qwen 3 235B / Qwen 3.5 397B | Apache 2.0 | > 200B total |

---

## Contributing

Found an error? Missing a model? Open an issue or submit a PR.

Sources: [HuggingFace](https://huggingface.co), [Papers With Code](https://paperswithcode.com), official model repos and papers.

---

## License

This list is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

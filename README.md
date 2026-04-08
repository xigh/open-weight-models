# Open Weight Models

A curated list of open-weight AI models with commercially exploitable licenses, verified benchmarks, and no geographic restrictions. Built to decide which models to support in [herbert-rs](https://github.com/xigh/herbert-rs), a local LLM inference engine in Rust and hand-written assembly.

**Selection criteria:**

1. **Commercially exploitable license**, no geographic restriction (EU ok)
2. **Total size < 200B** parameters
3. **Released after April 2024**

This excludes Llama 4 multimodal (EU exclusion), Qwen 3.6 Plus (closed-source), full DeepSeek V3/R1 (671B), and others. Note: Llama text-only models (3.3 70B, 3.2 1B/3B) are EU-exploitable. See [Rejected models](#rejected-models) for details.

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
  - [Vision / Multimodal](#vision--multimodal)
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
| [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Meta | 70B | 70B | Dense | 128K | Llama Community (EU OK) | MMLU 86.0, HumanEval 88.4, MATH 77.0 |
| [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) | Shanghai AI Lab | 78B | 78B | Dense | -- | Apache 2.0 | MMMU 72.2, SOTA open-source VLM |

### Code

| Model | SWE-bench | Codeforces | Active | License |
|-------|-----------|-----------|--------|---------|
| *[Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) (closed)* | *80.8%* | *--* | *--* | *--* |
| *[Gemini 3.1 Pro](https://deepmind.google/models/gemini/pro/) (closed)* | *80.6%* | *--* | *--* | *--* |
| *[GPT-5.4](https://openai.com/index/introducing-gpt-5-4/) (closed)* | *~80%* | *--* | *--* | *--* |
| [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | **74.4%** | -- | 11B | Apache 2.0 |
| [Devstral 2](https://huggingface.co/mistralai/Devstral-2) | 72.2% | -- | ~12B | MIT modified |
| [Qwen3-Coder-Next 80B-A3B](https://huggingface.co/Qwen/Qwen3-Coder-Next-80B-A3B) | 70.6% | -- | 3B | Apache 2.0 |
| [Qwen2.5-Coder-32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) | 69.6% | -- | 32B | Apache 2.0 |
| [Devstral Small 2](https://huggingface.co/mistralai/Devstral-Small-2-2505) | 68.0% | -- | 24B | Apache 2.0 |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 62.4% | **2622** | 5.1B | Apache 2.0 |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | -- | 2150 | 31B | Apache 2.0 |

> [SWE-bench](https://www.swebench.com/) = real bugs in real GitHub repos (Django, Flask, scikit-learn). 500 human-validated issues. [Codeforces](https://codeforces.com/) = algorithmic competition, ELO-scored like chess. Different skills: fixing a codebase vs solving a puzzle.

### Reasoning

#### [GPQA Diamond](https://arxiv.org/abs/2311.12022) (198 questions)

Graduate-level questions in physics, chemistry, biology. Designed to be unsolvable by Google search. Experts reach 65%, non-experts 34%. The most discriminating reasoning benchmark available.

| Model | GPQA | Active |
|-------|------|--------|
| *[Gemini 3.1 Pro](https://deepmind.google/models/gemini/pro/) (closed)* | *94.3* | *--* |
| *[GPT-5.4](https://openai.com/index/introducing-gpt-5-4/) (closed)* | *92.8* | *--* |
| *[Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) (closed)* | *91.3* | *--* |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | 84.3 | 31B |
| [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) | 82.3 | 3.8B |
| [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 81.7 | 9B |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 80.9 | 5.1B |
| [GLM-4.5-Air](https://huggingface.co/zai-org/GLM-4.5) | 75.0 | 12B |
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 73.0 | 3.5B |
| [Mistral Small 4](https://huggingface.co/mistralai/Mistral-Small-4-128K) | 71.2 | 6B |
| [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | 50.5 | 70B |

#### Math ([AIME](https://en.wikipedia.org/wiki/American_Invitational_Mathematics_Examination), 15 problems/year)

Competition-level math requiring creativity and multi-step reasoning. Each year's edition is different and harder. Only compare within the same version.

| Model | AIME | Conditions | Active |
|-------|------|-----------|--------|
| *[GPT-5.4](https://openai.com/index/introducing-gpt-5-4/) (closed)* | *~100%* | *2025* | *--* |
| *[Claude Opus 4.6](https://www.anthropic.com/news/claude-opus-4-6) (closed)* | *~98%* | *2025* | *--* |
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
| [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) | 3B | ~2 GB | Best 3B, AIME 36.7%, /think mode, 64K ctx | Apache 2.0 |
| [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct) | 1.7B | ~1 GB | 11T tokens, data-centric | Apache 2.0 |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | 360M | < 1 GB | 4T tokens | Apache 2.0 |
| [SmolLM2-135M](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) | 135M | < 1 GB | Ultra-compact, few MB quantized | Apache 2.0 |
| [Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) | 2.3B | ~4 GB | Multimodal + audio | Apache 2.0 |
| [Gemma 4 E4B](https://huggingface.co/google/gemma-4-E4B-it) | 4.5B | ~6 GB | Multimodal + audio | Apache 2.0 |
| [Phi-4-mini](https://huggingface.co/microsoft/phi-4) | 3.8B | ~2 GB | MATH-500 92.5% | MIT |
| [Phi-4-multimodal](https://huggingface.co/microsoft/phi-4) | 5.6B | ~3 GB | Text + image + audio | MIT |
| [Ministral 3B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 3B | ~2 GB | Vision + reasoning, 256K ctx | Apache 2.0 |
| [Ministral 8B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 8B | ~5 GB | AIME 78.7%, vision | Apache 2.0 |
| [Ministral 14B](https://huggingface.co/mistralai/Ministral-3-14B-Reasoning-2512) | 14B | ~8 GB | AIME 85%, vision, 256K ctx | Apache 2.0 |
| [LFM2.5-1.2B](https://huggingface.co/LiquidAI) | 1.2B | ~1 GB | IFBench 47.3 (2x Qwen3-1.7B), thinking, vision, audio | LFM Open v1.0 |
| [Llama 3.2 1B/3B](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 1-3B | < 2 GB | 128K ctx, edge/mobile, **EU OK** (text-only) | Llama Community |
| [InternLM3-8B](https://huggingface.co/internlm/internlm3-8b-instruct) | 8B | ~5 GB | Thinking mode, 4T tokens (75% less training) | Apache 2.0 |
| [InternVL3-1B→38B](https://huggingface.co/OpenGVLab/InternVL3-8B) | 1-38B | 1-20 GB | Vision SOTA, full range edge→server | Apache 2.0 |
| [Chocolatine-2-4B-DPO](https://huggingface.co/jpacifico/Chocolatine-2-4B-Instruct-DPO-v2.1) | 4B | ~2.5 GB | French-optimized DPO fine-tune of Qwen3-4B, 262K ctx, no `<think>` | Apache 2.0 |

> SmolLM3-3B beats all other 3B models and competes with 4B models (Qwen3-4B, Gemma3-4B). Data quality matters more than model size: SmolLM2-1.7B trained on 11T tokens beats larger models trained on less data.

> **Chocolatine-2-4B** (Jonathan Pacifico) is a DPO fine-tune of Qwen3-4B-Instruct-2507 on French preference datasets (Compar:IA from the French Ministry of Culture + French-ORCA), merged with TIES. Gains on every French benchmark tested (GPQA-FR, French MMLU, French Bench, FR-MT-Bench) without degrading English performance. One of the rare French-focused open-weight models built by an individual contributor rather than a lab.

### Long context

| Model | Max ctx | RULER 1M | Architecture | Active | License |
|-------|---------|---------|-------------|--------|---------|
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1M | **86.3%** | Mamba/MoE | 3.5B | Nemotron OML |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 1M | -- | Mamba/MoE | 12B | Nemotron OML |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | 256K | -- | SSM+Transformer/MoE | 12B | Jamba OML |

> [RULER](https://arxiv.org/abs/2404.06654) ([GitHub](https://github.com/NVIDIA/RULER)) tests retrieval in long contexts with multiple needles, multi-hop tracing, and aggregation. Parametric by length (4K to 1M). Many models claim "1M context" without publishing RULER scores at that length. Without measurement, it's marketing.

### Alternative architectures

Non-Transformer or hybrid models.

| Model | Architecture | Active | Key metric | License |
|-------|-------------|--------|-----------|---------|
| [Granite 4.0](https://huggingface.co/ibm-granite/granite-4.0-h-small) | 90% Mamba-2 / 10% Attention | 3-9B | 70% memory reduction, 2x speed | Apache 2.0 |
| [LFM2/2.5](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | Convolutions + grouped attention | 2.3B | 112 tok/s CPU, 2x Qwen3. LFM2.5: vision, audio, thinking | LFM Open v1.0 |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | Mamba + Transformer + MoE | 12B | 2.5x Transformer speed | Jamba OML |

---

## Specialized

### Theorem provers (Lean 4)

[miniF2F](https://arxiv.org/abs/2109.00110) ([GitHub](https://github.com/facebookresearch/miniF2F)): 488 formal Olympiad-level math problems. Proofs are compiler-verified: either correct or rejected. Zero hallucination possible on mathematical correctness.

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

[ScreenSpot](https://arxiv.org/abs/2504.07981) ([GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)): 1,200+ instructions across desktop, mobile, web. Tests if the model can locate the right UI element from a natural language instruction.

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

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) ([GitHub](https://github.com/ShishirPatil/gorilla)): Berkeley Function Calling Leaderboard. Tests function/tool calling accuracy: correct names, parameters, types. V4 adds web search and memory.

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

### Vision / Multimodal

| Model | MMMU | Active | Key feature | License |
|-------|------|--------|------------|---------|
| [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) | **72.2** | 78B | SOTA open-source VLM, custom InternViT | Apache 2.0 |
| [InternVL3-1B→38B](https://huggingface.co/OpenGVLab/InternVL3-8B) | -- | 1-38B | Full range edge→server | Apache 2.0 |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | Pro 76.9 | 31B | Text + image + video | Apache 2.0 |
| [Gemma 4 E2B/E4B](https://huggingface.co/google/gemma-4-E2B-it) | -- | 2.3-4.5B | Multimodal + audio, edge | Apache 2.0 |
| [Qwen2.5-VL-7B](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) | -- | 7B | Computer/phone use, DocVQA 95.7 | Apache 2.0 |

> InternVL3-78B (72.2 MMMU) is on par with GPT-4o on multimodal. The InternViT encoder (300M–6B) is trained jointly with the LLM — not bolted on after the fact.

---

## Observations

Patterns observed across 60+ models. Not definitive truths.

### Architecture

- **Dense dies above 35B.** Every model > 35B released in 2025-2026 is MoE. The last competitive denses at this scale are Llama 3.3 70B and DeepSeek R1-Distill-70B, both from late 2024/early 2025. No new dense > 35B has appeared since.

- **Parameter count is no longer the determining factor.** Qwen3.5-9B (9B) beats GPT-OSS-120B (5.1B active, 117B total) on GPQA Diamond.

- **The 40-79B segment is thinning out.** New models tend to jump from ~35B to ~120B total via MoE. The segment still has strong entries (Llama 3.3 70B, Qwen 2.5-72B, InternVL3-78B, R1-Distill-70B, Jamba 1.6 Mini 52B), but no new entrants in 2025-2026.

- **InternVL3 is the best open-source VLM nobody was talking about.** InternVL3-78B (Shanghai AI Lab) reaches 72.2 MMMU under Apache 2.0 — on par with GPT-4o. InternLM3-8B achieves SOTA with 75% fewer training tokens (4T vs 15-18T). Less press than Alibaba, comparable results.

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

- **Llama 4 excludes the EU** for multimodal models. But text-only Llama (3.3 70B, 3.2 1B/3B) is EU-exploitable — the exclusion only applies to multimodal.

- **"Open-weight" is more nuanced than "open-source".** Llama is technically open-weight but with geographic restrictions on multimodal. Always check the fine print.

---

## Benchmarks reference

What each benchmark measures, how many questions it has, and where to find more.

### Reasoning & Knowledge

- **[GPQA Diamond](https://arxiv.org/abs/2311.12022)** (198 questions) — Graduate-level questions in physics, chemistry, biology. Designed to be unsolvable by Google search. Experts reach 65%, non-experts 34%. The most discriminating reasoning benchmark.

- **[MMLU-Pro](https://arxiv.org/abs/2406.01574)** (12K+ questions) — Hardened version of MMLU: 10 choices instead of 4, requires chain-of-thought reasoning. 14 domains. Drops accuracy 16-33% vs MMLU. Published at NeurIPS 2024.

### Math

- **AIME** (15 problems/year) — American Invitational Mathematics Examination. Competition-level math requiring creativity and multi-step reasoning. Each year's edition is harder. Only compare within the same version (2024/2025/2026).

- **MATH-500** (500 problems) — Diverse math problems (algebra, geometry, combinatorics, number theory). Good general math evaluation but easier to saturate than AIME.

### Code

- **[SWE-bench Verified](https://www.swebench.com/)** (500 issues) — Real bugs from GitHub repos (Django, Flask, scikit-learn). The model must understand the codebase, find the bug, and produce a working patch. Human-validated by OpenAI. [Paper](https://arxiv.org/abs/2310.06770)

- **Codeforces** (ELO system) — Algorithmic competition performance, scored like chess ELO. Measures pure algorithmic skill, not real-world coding. Different skill from SWE-bench.

- **[LiveCodeBench](https://livecodebench.github.io/)** (rotating, 700+) — Fresh competitive programming problems collected after model training cutoffs. Eliminates data contamination. Problems from LeetCode, AtCoder, Codeforces. [GitHub](https://github.com/LiveCodeBench/LiveCodeBench)

### Long context

- **[RULER](https://arxiv.org/abs/2404.06654)** (parametric) — Sophisticated "needle in a haystack" with multiple needles, multi-hop tracing, and aggregation. Tests at different lengths (4K to 1M). By NVIDIA. Many models claiming 1M context fail above 32K. [GitHub](https://github.com/NVIDIA/RULER)

### Agents & Tools

- **[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html)** (2K+) — Berkeley Function Calling Leaderboard. Tests function/tool calling accuracy: correct names, parameters, types. V4 adds web search and memory. By UC Berkeley. [GitHub](https://github.com/ShishirPatil/gorilla)

### Theorem proving

- **[miniF2F](https://arxiv.org/abs/2109.00110)** (488 problems) — Formal Olympiad-level math problems in Lean 4 (also Isabelle, HOL Light). Covers AMC, AIME, IMO, and university math. Proofs are compiler-verified: either correct or rejected. Zero hallucination possible. [GitHub](https://github.com/facebookresearch/miniF2F)

### GUI

- **[ScreenSpot](https://arxiv.org/abs/2504.07981)** (1.2K+ instructions) — GUI element grounding across desktop, mobile, and web. Tests if the model can locate the right UI element from a natural language instruction. [GitHub](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding)

---

## Licenses

| License | Models | Commercial | EU | Patent grant | OSI |
|---------|--------|-----------|-----|-------------|-----|
| Apache 2.0 | Gemma 4, Qwen 3/3.5, GPT-OSS, Ministral, Step-3.5-Flash | Yes | Yes | Yes | Yes |
| MIT | GLM-4.5-Air, DeepSeek R1-Distill, Phi-4 | Yes | Yes | No (implicit) | Yes |
| Nemotron OML | Nemotron 3 Nano/Super | Yes | Yes | Yes | No |
| Jamba OML | Jamba 1.6 | Yes | Yes | -- | No |
| Llama Community | Llama 3.3 70B, Llama 3.2 1B/3B (text-only) | Yes | **Yes** (text-only) | -- | No |
| LFM Open v1.0 | LFM2, LFM2.5 | Yes (< $10M) | Yes | -- | No |

---

## How to choose

| Constraint | Recommendation |
|-----------|---------------|
| Smartphone / edge (< 4 GB) | SmolLM3-3B, SmolLM2-135M/360M/1.7B, Gemma 4 E2B, Phi-4-mini, Ministral 3B, LFM2.5-1.2B, Llama 3.2 1B/3B |
| Laptop 16 GB | GPT-OSS-20B, Ministral 14B, Gemma 4 26B-A4B |
| Desktop 24 GB | Gemma 4 31B, DeepSeek R1-Distill-32B, Devstral Small 2 |
| Desktop 48+ GB (dense 70B) | Llama 3.3 70B (MMLU 86.0, EU OK), InternVL3-78B (vision) |
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
| Llama 4 (Maverick, Scout) | Llama Community License | EU exclusion (multimodal) |
| Llama 3.2 Vision 11B/90B | Llama Community License | EU exclusion (multimodal) |
| Llama-Nemotron-Super-49B | Llama 3.3 License | Inherits EU exclusion (multimodal base) |
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

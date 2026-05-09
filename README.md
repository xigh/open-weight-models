# Open Weight Models

A curated list of open-weight AI models with commercially exploitable licenses, verified benchmarks, and no geographic restrictions. Built to decide which models to support in [herbert-rs](https://github.com/xigh/herbert-rs), a local LLM inference engine in Rust and hand-written assembly.

**Selection criteria:**

1. **Commercially exploitable license**, no geographic restriction (EU ok)
2. **VRAM Q4 ≤ 128 GB** (main) — what fits Q4-quantized on a single high-end workstation. An **extended tier** (128 < Q4 ≤ 256 GB) flags models that require a Mac Studio M3 Ultra or multi-GPU server.
3. **Released after April 2024**

This excludes Llama 4 multimodal (EU exclusion), Qwen 3.6 Plus (closed-source), DeepSeek V3/R1 full (671B → ~370 GB Q4, beyond 256 GB), and others. Note: Llama text-only models (3.3 70B, 3.2 1B/3B) are EU-exploitable. See [Rejected models](#rejected-models) for details.

> Maintained by [Philippe Anel](https://philippe-anel.fr). Last updated: May 2026.

> **v3 additions (May 2026)** — Selection criterion switched from "< 200B params" to "VRAM Q4 ≤ 128 GB" + extended 256 GB tier (better proxy for what's actually runnable on consumer/prosumer hardware). Generalists: Ling-2.6-flash 104B/A7.4B (Ant), Mistral Medium 3.5 128B (🔴 Modified MIT), MiniMax M2.7 230B/A10B. Extended tier: DeepSeek-V4-Flash 284B/A13B (1M ctx native, FP4+FP8). Alternative architectures: ZAYA1-8B (Zyphra), Kimi-Linear 48B/A3B (Moonshot, KDA hybrid), Bonsai-8B (1-bit end-to-end, 1.15 GB). Vision/Multimodal: Nemotron 3 Nano Omni 30B-A3B, DeepSeek-OCR + DeepSeek-OCR-2. Compact/Edge: LFM2.5-VL-450M (vision edge). Theorem provers: SGS algorithm (Stanford, 7B beats 671B pass@4 on D3k). New license row: 🔴 **Modified MIT** with explicit warning.
>
> **v2 additions (April 2026)** — Generalists: GLM-4.7-Flash, Hermes 4-70B. Code: NousCoder-14B, OmniCoder-9B (new LCB/Terminal-Bench subsection). Compact/Edge: Pleias-RAG-1B, Pleias-3B. Reasoning/Math: Qwen2.5-Math-72B (historical). Alternative architectures: URM. Decentralized training: Hermes 4.3-36B-Psyche. Theorem provers: Nomos 1 (natural-language track).

---

## Table of Contents

- [LLMs](#llms)
  - [Generalists](#generalists)
    - [Extended tier (128 < Q4 ≤ 256 GB)](#extended-tier-128--q4--256-gb)
  - [Code](#code)
    - [Code — LiveCodeBench & Terminal-Bench](#code--livecodebench--terminal-bench)
  - [Reasoning](#reasoning)
  - [Compact / Edge](#compact--edge)
  - [Long context](#long-context)
  - [Alternative architectures](#alternative-architectures)
  - [Decentralized training](#decentralized-training)
- [Specialized](#specialized)
  - [Theorem provers (Lean 4)](#theorem-provers-lean-4)
    - [Natural-language provers (not Lean 4)](#natural-language-provers-not-lean-4)
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
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | Zhipu AI | 3B | 30B | MoE (MLA) | 200K | MIT | SWE-bench 59.2, AIME25 91.6, GPQA 75.2 |
| [Ling-2.6-flash](https://huggingface.co/inclusionAI/Ling-2.6-flash) | Ant Group | 7.4B | 104B | MoE (hybrid linear attn 1:7 MLA+Lightning) | 262K | MIT | Token-efficient agent (~15M tokens on full AA suite vs 40-100M for long-reasoners) |
| [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) | Alibaba | 32B | 32B | Dense | 128K | Apache 2.0 | AIME ~80%, reasoning RL |
| [DeepSeek R1-Distill-32B](https://huggingface.co/deepseek-ai/DeepSeek-R1) | DeepSeek | 32B | 32B | Dense | 128K | MIT | Beats o1-mini |
| [Step-3.5-Flash](https://huggingface.co/stepfun-ai/Step-3.5-Flash) | StepFun | 11B | 196B | MoE | 262K | Apache 2.0 | SWE-bench 74.4%, 350 tok/s |
| [Llama 3.3 70B](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | Meta | 70B | 70B | Dense | 128K | Llama Community (EU OK) | MMLU 86.0, HumanEval 88.4, MATH 77.0 |
| [Hermes 4-70B](https://huggingface.co/NousResearch/Hermes-4-70B) | Nous Research | 70B | 70B | Dense | 128K | Llama Community (EU OK) | SOTA RefusalBench, hybrid reasoning, tool calling |
| [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) | Shanghai AI Lab | 78B | 78B | Dense | -- | Apache 2.0 | MMMU 72.2, SOTA open-source VLM |
| [Mistral Medium 3.5 128B](https://huggingface.co/mistralai/Mistral-Medium-3.5-128B) | Mistral AI | 128B | 128B | Dense + Pixtral vision | 256K | 🔴 **Modified MIT** (revenue cap) | First Mistral merged flagship: Medium 3.1 + Magistral + Devstral 2 unified, configurable `reasoning_effort` |
| [MiniMax M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) | MiniMax AI | 10B | 230B | MoE (256 experts, 8 active, 4.3% ratio) | ~200K | MIT *(verify on HF)* | Agentic workflows alt to Claude Opus 4.6 / GPT-5.3-Codex, IQ1_M @ 60.7 GB |

> **🔴 License warning — Modified MIT (revenue/MAU caps).** Mistral Medium 3.5 falls under a Mistral Open License variant with a revenue threshold; MiniMax M2.7 and Kimi K2.5 historically shipped with similar caps (100M MAU for Kimi). They are **listed for completeness** but you must read the actual license before any commercial deployment — these are **not interchangeable with Apache 2.0/MIT**. The revenue/MAU clauses can flip a free model into a paid one once your product takes off.

> **Mistral Medium 3.5** (Apr/May 2026) is the **first merged flagship** from Mistral: a single set of weights unifying what used to be three distinct models — Medium 3.1 (instruct), Magistral (reasoning), Devstral 2 (coding agent). Behavior switches via `reasoning_effort` per request (`none` / `high`). Replaces Medium 3.1 + Magistral in Le Chat and Devstral 2 in Vibe CLI. 88-layer dense (no MoE), Pixtral vision tower trained from scratch.

> **MiniMax M2.7** (Apr 2026 open-weight release) pushes the **active/total ratio** to 4.3% (10B/230B), targeting agentic long-running workflows (coding, multi-step troubleshooting, document editing). Positioned as open-weight alternative to Claude Opus 4.6 / GPT-5.3-Codex with IQ1_M weights at 60.7 GB making 230B practically deployable on a single workstation.

#### Extended tier (128 < Q4 ≤ 256 GB)

Models that exceed the 128 GB Q4 main cap but fit a 256 GB workstation (Mac Studio M3 Ultra max, multi-GPU server).

| Model | Publisher | Active | Total | Arch | Ctx | License | Key scores |
|-------|-----------|--------|-------|------|-----|---------|-----------|
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) | DeepSeek | 13B | 284B | MoE (hybrid CSA+HCA, mHC, Muon optimizer) | **1M** native | MIT | First < 200B-active LLM with native 1M ctx, FP4+FP8 mixed, 32T pre-train, 27% FLOPs / 10% KV cache vs V3.2 |

> **DeepSeek-V4-Flash** (May 2026) is the small sibling of V4-Pro (1.6T/49B). Q4 ≈ 156 GB. Three inference modes integrated in the chat template: non-think, think-high, think-max (recommended at ≥ 384K context). The architecture introduces three new ideas — hybrid attention (CSA + HCA), multi-head computation (mHC), and the Muon optimizer — pushing the efficiency frontier rather than the parameter frontier.

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
| [GLM-4.7-Flash](https://huggingface.co/zai-org/GLM-4.7-Flash) | 59.2% | -- | 3B | MIT |
| [GPT-OSS-120B](https://huggingface.co/openai/GPT-OSS-120B) | 62.4% | **2622** | 5.1B | Apache 2.0 |
| [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) | -- | 2150 | 31B | Apache 2.0 |

> [SWE-bench](https://www.swebench.com/) = real bugs in real GitHub repos (Django, Flask, scikit-learn). 500 human-validated issues. [Codeforces](https://codeforces.com/) = algorithmic competition, ELO-scored like chess. Different skills: fixing a codebase vs solving a puzzle.

#### Code — LiveCodeBench & Terminal-Bench

Specialized coders measured on benchmarks other than SWE-bench.

| Model | LiveCodeBench v6 | Terminal-Bench 2.0 | Active | License |
|-------|-----------------|--------------------|--------|---------|
| [OmniCoder-9B](https://huggingface.co/Tesslate/OmniCoder-9B) | -- | **23.6%** | 9.4B | Apache 2.0 |
| [NousCoder-14B](https://huggingface.co/NousResearch/NousCoder-14B) | **67.87%** | -- | 14.8B | Apache 2.0 |
| Qwen3.5-9B (baseline) | 60.79% | 14.6% | 9B | Apache 2.0 |

> **[LiveCodeBench](https://livecodebench.github.io/)** (rotating ≈700 problems from LeetCode/AtCoder/Codeforces, collected after model cutoffs) measures fresh competitive programming, vs SWE-bench (fixing real-world bugs) and Codeforces ELO (pure algorithms). **[Terminal-Bench 2.0](https://www.tbench.ai/)** measures agentic coding skills (read-before-write, LSP responsiveness, minimal diffs).

> OmniCoder-9B is a **LoRA agentic fine-tune** of Qwen3.5-9B on 425K Claude Opus 4.6 / GPT-5.4 / Gemini 3.1 Pro trajectories — +61% relative on Terminal-Bench vs base. NousCoder-14B is a **pure-RL fine-tune** of Qwen3-14B (+7.08 pts on LCB v6, no SFT). Same 9-14B class, opposite methods.

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
| [Qwen2.5-Math-72B](https://huggingface.co/Qwen/Qwen2.5-Math-72B-Instruct) | 40.0% | 2024, TIR (Python) | 72B |

> AIME versions (2024/2025/2026) are not comparable. Each year is harder.

> **Qwen2.5-Math (Sep 2024)** is the first open model to make real AIME progress (12/30 on AIME 2024 — 6× GPT-4 Turbo at the time). Since surpassed by generalists (Nemotron, GPT-OSS 96%+). Historical reference, still useful for its **TIR mode** (Tool-Integrated Reasoning with Python interpreter) which eliminates arithmetic errors. Qwen License, not Apache — commercial OK but check terms.

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
| [Pleias-RAG-1B](https://huggingface.co/PleIAs/Pleias-RAG-1B) | 1.2B | ~1 GB | **100% public-domain training data**, native citation with literal quotes, EU multilingual | Apache 2.0 |
| [Pleias-RAG-350M](https://huggingface.co/PleIAs/Pleias-RAG-350M) | 350M | < 1 GB | Same as Pleias-RAG-1B, ultra-compact | Apache 2.0 |
| [Baguettotron](https://huggingface.co/PleIAs/Baguettotron) | 0.3B | < 1 GB | Latest Pleias base (Dec 2025), French-focused SLM | Apache 2.0 |
| [LFM2.5-VL-450M](https://huggingface.co/LiquidAI) | 450M | < 1 GB | **Vision edge**: SigLIP2 + 512×512 native, object detection, WebGPU | LFM Open v1.0 |
| [Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo) | 8B | **1.15 GB** | 1-bit Qwen3-8B fine-tune, CUDA/Metal/CPU/Android/iPhone | Apache 2.0 |

> SmolLM3-3B beats all other 3B models and competes with 4B models (Qwen3-4B, Gemma3-4B). Data quality matters more than model size: SmolLM2-1.7B trained on 11T tokens beats larger models trained on less data.

<a id="chocolatine"></a>

> **Chocolatine-2-4B** (Jonathan Pacifico) is a DPO fine-tune of Qwen3-4B-Instruct-2507 on French preference datasets (Compar:IA from the French Ministry of Culture + French-ORCA), merged with TIES. Gains on every French benchmark tested (GPQA-FR, French MMLU, French Bench, FR-MT-Bench) without degrading English performance. One of the rare French-focused open-weight models built by an individual contributor rather than a lab.

> **Pleias** (Paris-based lab, partners with NVIDIA and Mozilla Builders) trains exclusively on **public-domain or CC-licensed data** (Common Corpus, 2T tokens). Raw benchmark scores trail Qwen/Gemma of equivalent size, but the trade-off is unique: **zero copyright ambiguity** (EU AI Act / GDPR friendly), strong EU multilingual (FR, DE, IT, ES, NL, PL), and the Pleias-RAG variants emit **literal-quote citations** natively. Positioned for regulated sectors (public, legal, press, education) where data provenance matters more than peak scores.

### Long context

| Model | Max ctx | RULER 1M | Architecture | Active | License |
|-------|---------|---------|-------------|--------|---------|
| [Nemotron 3 Nano](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) | 1M | **86.3%** | Mamba/MoE | 3.5B | Nemotron OML |
| [Nemotron 3 Super](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16) | 1M | -- | Mamba/MoE | 12B | Nemotron OML |
| [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) *(extended tier)* | **1M native** | -- | MoE (CSA+HCA hybrid) | 13B | MIT |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | 256K | -- | SSM+Transformer/MoE | 12B | Jamba OML |

> [RULER](https://arxiv.org/abs/2404.06654) ([GitHub](https://github.com/NVIDIA/RULER)) tests retrieval in long contexts with multiple needles, multi-hop tracing, and aggregation. Parametric by length (4K to 1M). Many models claim "1M context" without publishing RULER scores at that length. Without measurement, it's marketing.

### Alternative architectures

Non-Transformer or hybrid models.

| Model | Architecture | Active | Key metric | License |
|-------|-------------|--------|-----------|---------|
| [Granite 4.0](https://huggingface.co/ibm-granite/granite-4.0-h-small) | 90% Mamba-2 / 10% Attention | 3-9B | 70% memory reduction, 2x speed | Apache 2.0 |
| [LFM2/2.5](https://huggingface.co/LiquidAI/LFM2-24B-A2B) | Convolutions + grouped attention | 2.3B | 112 tok/s CPU, 2x Qwen3. LFM2.5: vision, audio, thinking | LFM Open v1.0 |
| [Jamba 1.6 Mini](https://huggingface.co/ai21labs/Jamba-1.6-Mini) | Mamba + Transformer + MoE | 12B | 2.5x Transformer speed | Jamba OML |
| [URM](https://github.com/UbiquantAI/URM) | Recursive Universal Transformer (ConvSwiGLU + TBPTL) | 4× params (tiny) | ARC-AGI 1: **53.8%**, Sudoku 77.6% | Open-source (research) |
| [ZAYA1-8B](https://huggingface.co/Zyphra/ZAYA1-8B) | Hybrid Mamba + Compressed Cross Attention (CCA) + MoD + EDA | 760M / 8.4B (**9% active**) | On-device deployable, test-time-compute friendly, 128K ctx | Apache 2.0 |
| [Kimi-Linear-48B-A3B](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct) | MoE hybrid: 3 KDA (linear) layers per 1 MLA (global) | 3B / 48B | **1M context**, 5.7T tokens, demonstrates linear attention can match full attention | MIT |
| [Bonsai-8B](https://github.com/PrismML-Eng/Bonsai-demo) | Qwen3-8B fine-tuned at **1-bit end-to-end** (GGUF Q1_0), all projections + LM head 1-bit | 8.19B | **1.15 GB on disk** (14.2× FP16), runs on CPU/Android/iPhone | Apache 2.0 |

> **URM** (Ubiquant, Dec 2025) loops its 4 layers 12× instead of stacking 48 distinct layers. With **4× parameters** it reaches 53.8% on ARC-AGI 1 where a vanilla Transformer with 32× parameters stays under 40%. Key claim of the paper: **the FFN, not attention, is the source of reasoning** — counterintuitive given the community's focus on attention variants. Research model, not a production LLM, but architecturally interesting for future LLM designs. See [arXiv:2512.14693](https://arxiv.org/abs/2512.14693).

> **ZAYA1-8B** (Zyphra, May 2026) is a Zamba-2 successor: 80 layers mixing SSM-Mamba and attention, with **Compressed Cross Attention (CCA)** plus **Mixture-of-Depths (MoD)** and **Expert Decision Attention (EDA)** on top of 16 top-1 experts. The angle is *intelligence per active parameter*: 760M actifs gives sub-1B inference cost while keeping 8B-class capacity. Positioned for on-device + thinking-mode workflows where compute scales with active params, not totals. Tech report on [zyphra.com/zaya1-8b-technical-report](https://www.zyphra.com/zaya1-8b-technical-report).

> **Kimi-Linear** (Moonshot, Oct 2025, [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)) is Moonshot's open research vehicle for **linear attention** outside the closed K2 family. The architecture is a 3:1 ratio of **KDA (Kimi Delta Attention, linear)** to **MLA (full attention, global)** layers. The point isn't frontier performance — it's the demonstration that linear attention can **match full attention across short, long, and RL-style regimes** while reducing memory cost. Useful baseline for engine work like herbert-rs.

> **Bonsai-8B** (Prism ML, Mar 2026) is a **1-bit end-to-end** fine-tune of Qwen3-8B: every projection + the LM head quantized to 1 bit (GGUF Q1_0), shrinking the deployed model to **1.15 GB**. Direct competitor to BitNet, but trained as a *fine-tune* rather than natively 1.58-bit from scratch. Runs on CUDA, Metal, Android, CPU, and iPhone (via Locally AI). The radical end of the quantization spectrum — accept the quality drop in exchange for ubiquity.

### Decentralized training

Models pre-trained outside traditional data centers, using distributed peer-to-peer or blockchain-coordinated networks. The story is the training method, not the model quality.

| Model | Method | Size | Tokens | Architecture | License |
|-------|--------|------|--------|--------------|---------|
| [Covenant-72B](https://arxiv.org/abs/2603.08163) | Permissionless P2P, SparseLoCo optimizer, Bittensor blockchain (Subnet 3) | 72B dense | 1.1T (+14.8B SFT) | LLaMA-3 style, GQA, 80 layers, d=8192, 64 heads, 8 KV heads, RoPE 500K, ctx 2048→8192 | Apache 2.0 (checkpoints) |
| [Hermes 4.3-36B-Psyche](https://huggingface.co/NousResearch/Hermes-4.3-36B) | Internet-decentralized fine-tuning via [Psyche](https://psyche.network) | 36B dense | — (post-training on Seed-36B) | ByteDance Seed-36B base, Llama-3 chat template, hybrid `<think>` mode | Apache 2.0 |

**Pre-training benchmarks (0-shot)** vs other dense baselines :

| Benchmark | Covenant-72B | LLaMA-2-70B (centralized) | LLM360 K2 (65B, centralized) | INTELLECT-1 (10B, P2P) |
|-----------|-------------:|--------------------------:|-----------------------------:|-----------------------:|
| ARC-Challenge | **56.8** | 57.4 | 53.8 | 44.8 |
| ARC-Easy | **80.9** | 79.6 | 76.0 | 71.8 |
| PIQA | 81.6 | 82.6 | 82.5 | 77.4 |
| OpenBookQA | 44.0 | 49.4 | 48.0 | 43.8 |
| HellaSwag | 80.6 | 84.3 | 82.9 | 70.3 |
| WinoGrande | 75.9 | 80.4 | 76.4 | 63.3 |
| MMLU | **67.1** | 65.6 | 65.5 | 32.7 |

**Covenant-72B-Chat (post-SFT)** vs other chat models :

| Benchmark | Covenant-72B-Chat | LLaMA-2-70B-Chat | K2-Chat (65B) |
|-----------|------------------:|-----------------:|---------------:|
| ARC-Challenge | 64.2 | 65.4 | 62.0 |
| MMLU | 67.4 | 63.1 | 67.9 |
| **IFEval** | **64.7** | 40.7 | 45.5 |
| **MATH** | **26.3** | 10.7 | 19.1 |
| MMLU-Pro | 40.9 | 35.2 | 45.4 |
| GSM8K | 63.9 | 52.2 | 79.0 |

> **Hermes 4.3-36B-Psyche** (Nous Research, Nov 2025) is a different point in the same space: not pre-training from scratch but **post-training decentralized over internet**. Built on ByteDance's Seed-36B, fine-tuned via Nous's [Psyche network](https://psyche.network), released under Apache 2.0. The Psyche variant **matches or beats the centralized 4.3-36B twin on every benchmark** (AIME25 69.3 vs 66.8, MMLU-Pro 80.7 vs 79.7) — decentralized post-training did not degrade quality. Complements Covenant: two different decentralization angles (pre-training at 72B / post-training at 36B).

> **Why Covenant matters**: Covenant-72B is the first proof-of-concept that 72B-scale pre-training is possible without data centers, with peers joining and leaving freely. Coordination via the Bittensor blockchain (Subnet 3), communication via SparseLoCo (146× compression vs dense gradients), peers running 8×B200 GPUs over commodity internet (500 Mb/s down, 110 Mb/s up). The model achieves **94.5% compute utilization** despite the network constraints, with an average of 16.9 contributing peers per round and 70+ unique peers over the run. On benchmarks, it beats LLaMA-2-70B on ARC-Challenge, ARC-Easy and MMLU (despite 1.8× fewer training tokens), and the chat variant has the best IFEval and MATH scores in its comparison group. It's the first credible alternative to the data-center duopoly for pre-training at 70B scale. Authors: Covenant AI + Mila. See [arXiv 2603.08163](https://arxiv.org/abs/2603.08163).

---

## Specialized

### Theorem provers (Lean 4)

[miniF2F](https://arxiv.org/abs/2109.00110) ([GitHub](https://github.com/facebookresearch/miniF2F)): 488 formal Olympiad-level math problems. Proofs are compiler-verified: either correct or rejected. Zero hallucination possible on mathematical correctness.

| Model | miniF2F | PutnamBench | Active | License |
|-------|---------|-----------|--------|---------|
| [BFS-Prover-V2-32B](https://huggingface.co/ByteDance-Seed/BFS-Prover-V2) | **95.0%** | -- | 32B | Apache 2.0 |
| [Goedel-Prover-V2-32B](https://huggingface.co/Goedel-LM/Goedel-Prover-V2-32B) | 90.4% | **#1** | 32B | Apache 2.0 |
| [DeepSeek-Prover-V2-7B](https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-7B) | 88.9% | -- | 7B | MIT |
| [DeepSeek-Prover-V2-7B + SGS](https://github.com/LukeBailey181/sgs) | -- | -- | 7B | MIT (model) / CC-BY-4.0 (paper) |
| [Leanstral](https://huggingface.co/mistralai/Leanstral) | -- | -- | 32B | Apache 2.0 |
| [Kimina-Prover-72B](https://huggingface.co/MoonshotAI/Kimina-Prover-Preview-Distill-72B) | 84.0% | -- | 72B | MIT |
| [Leanabell-Prover-V2-7B](https://huggingface.co/leanbell/Leanabell-Prover-V2-7B) | 78.2% | -- | 7B | Apache 2.0 |

> Lean 4 proofs are verified by the compiler. Either correct or rejected. Zero hallucination on mathematical correctness.

> The sweet spot is 32B: BFS-Prover (95%) and Goedel-V2 (90.4%) both beat the 72B Kimina (84%).

> **SGS — Self-Guided Self-Play** (Stanford, [arXiv:2604.20209](https://arxiv.org/abs/2604.20209), Apr 2026) is not a model but an RL self-play **algorithm** applied to DeepSeek-Prover-V2-7B. After 200 rounds and 6.3M generations, **the 7B fine-tune surpasses the pass@4 of DeepSeek-Prover-V2-671B on D3k** (3 323 Lean 4 problems from Goedel-Pset-V1). Caveat: D3k is the SGS run's own training-target set, not a held-out public benchmark like miniF2F or PutnamBench — the 7B-beats-671B headline is real for in-distribution problems, scope-restricted otherwise. Demonstrates that **well-tuned RL can collapse a 100× parameter gap** on a target dataset. Authors: Bailey, Wen, Dong, Hashimoto, Ma.

#### Natural-language provers (not Lean 4)

A parallel track: models that write proofs in natural English, not formal Lean 4. Not compiler-verified, but closer to how mathematicians actually work.

| Model | Benchmark | Active | Total | License |
|-------|-----------|--------|-------|---------|
| [Nomos 1](https://huggingface.co/NousResearch/nomos-1) | Putnam 2025: **87/120 (72.5%)** with Nomos Harness | ~3B | ~30B | Apache 2.0 |

> **Nomos 1** (Nous Research × Hillclimb AI, Dec 2025) is a Qwen3-30B-A3B-Thinking fine-tune specialized for **natural-language proof writing**, not Lean 4. On Putnam 2025 with the open-sourced [Nomos Reasoning Harness](https://github.com/NousResearch/nomos), it jumps from 24/120 (base) to 87/120 — a +63 point gain where the inference harness matters as much as the model. Complementary to the Lean provers above, which offer compiler-verification guarantees that natural-language proofs cannot.

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
| [Nemotron 3 Nano Omni 30B-A3B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B) | -- | 3B | **any-to-any** (text+audio+image+video → text), 256K ctx, Mamba-Transformer hybrid MoE | Nemotron OML |
| [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) | -- | 3.3B | **OCR specialist** — *Contexts Optical Compression*: encode long text as compressed image, feed books/papers as pixels not tokens | MIT |
| [DeepSeek-OCR-2](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2) | -- | 3.4B | OCR v2 with **Visual Causal Flow** (sequential reading order) | Apache 2.0 |

> InternVL3-78B (72.2 MMMU) is on par with GPT-4o on multimodal. The InternViT encoder (300M–6B) is trained jointly with the LLM — not bolted on after the fact.

> **Nemotron 3 Nano Omni** (NVIDIA, Apr 2026) extends the Nano family with native audio/video/image inputs. Targets enterprise document intelligence (contracts, SOW/MSA, finance), customer service (drive-thru order verification, delivery video OCR), GUI/browser/email agents, and dense video captioning. Stack of three components around the Nano LLM (vision encoder + audio encoder + LLM), not a monolithic any-to-any architecture. English-only. See [arXiv:2604.24954](https://arxiv.org/abs/2604.24954).

---

## Observations

Patterns observed across 60+ models. Not definitive truths.

### Architecture

- **Dense retreats above 35B, but doesn't die.** For generalists above 35B, MoE clearly dominates (GPT-OSS-120B, Mistral Small 4, Qwen3.5-122B-A10B, GLM-4.5-Air, Step-3.5-Flash, Nemotron 3 Super, all MoE). But dense survives where it has a structural advantage: Llama 3.3 70B (generalist), InternVL3-78B (vision), Kimina-Prover-72B (theorem proving), Qwen 2.5-72B (production NLP), Covenant-72B (decentralized training), DeepSeek R1-Distill-70B (distilled reasoning). Dense is becoming a specialization choice.

- **Parameter count is no longer the determining factor.** Qwen3.5-9B (9B) beats GPT-OSS-120B (5.1B active, 117B total) on GPQA Diamond.

- **The 40-79B segment is the dense survivors' refuge.** New models often jump from ~35B straight to ~120B total via MoE. But the 40-79B range is well populated by quality dense models (Llama 3.3 70B, InternVL3-78B, Kimina-Prover-72B, Qwen 2.5-72B, Covenant-72B, R1-Distill-70B, Jamba 1.6 Mini 52B). This is where dense resists, and where you find both solid generalists and specialists.

- **InternVL3 is the best open-source VLM nobody was talking about.** InternVL3-78B (Shanghai AI Lab) reaches 72.2 MMMU under Apache 2.0 — on par with GPT-4o. InternLM3-8B achieves SOTA with 75% fewer training tokens (4T vs 15-18T). Less press than Alibaba, comparable results.

- **Qwen is the de facto base model** for fine-tuning. BFS-Prover, Goedel-Prover, Kimina-Prover, most community distillations: all built on Qwen. The ResNet of LLMs.

- **Decentralized pre-training is no longer a toy.** Covenant-72B (Mar 2026) pre-trained a 72B dense LLaMA-3-style model over a permissionless blockchain network (Bittensor Subnet 3) on 1.1T tokens. It beats LLaMA-2-70B on ARC-Challenge, ARC-Easy and MMLU despite 1.8× fewer training tokens, with 94.5% compute utilization over commodity internet (500/110 Mb/s) and dynamic peer participation. The data-center duopoly for pre-training at 70B scale now has a credible alternative. SparseLoCo + 2-bit quantization gives 146× compression on gradient communication.

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
| Apache 2.0 | Gemma 4, Qwen 3/3.5, GPT-OSS, Ministral, Step-3.5-Flash, NousCoder, OmniCoder, Nomos 1, URM, ZAYA1, Bonsai, DeepSeek-OCR-2, Hermes 4.3-36B, Pleias (all variants), Baguettotron | Yes | Yes | Yes | Yes |
| MIT | GLM-4.5-Air, GLM-4.7-Flash, DeepSeek R1-Distill, DeepSeek-V4-Flash, DeepSeek-OCR (v1), Ling-2.6-flash, Kimi-Linear, MiniMax M2.7 *(verify on HF)*, Phi-4 | Yes | Yes | No (implicit) | Yes |
| 🔴 **Modified MIT** (revenue/MAU caps) | **Mistral Medium 3.5** (revenue cap), historically Kimi K2.5 (100M MAU), MiniMax M2.5 | Conditional | Conditional | -- | No |
| Nemotron OML | Nemotron 3 Nano/Super, Nemotron 3 Nano Omni | Yes | Yes | Yes | No |
| Jamba OML | Jamba 1.6 | Yes | Yes | -- | No |
| Llama Community | Llama 3.3 70B, Llama 3.2 1B/3B (text-only), Hermes 4-70B | Yes | **Yes** (text-only) | -- | No |
| LFM Open v1.0 | LFM2, LFM2.5, LFM2.5-VL | Yes (< $10M) | Yes | -- | No |
| Qwen License | Qwen2.5-Math | Yes | Yes | -- | No |

---

## How to choose

| Constraint | Recommendation |
|-----------|---------------|
| Smartphone / edge (< 4 GB) | SmolLM3-3B, SmolLM2-135M/360M/1.7B, Gemma 4 E2B, Phi-4-mini, Ministral 3B, LFM2.5-1.2B, Llama 3.2 1B/3B |
| Laptop 16 GB | GPT-OSS-20B, Ministral 14B, Gemma 4 26B-A4B |
| Desktop 24 GB | Gemma 4 31B, DeepSeek R1-Distill-32B, Devstral Small 2, **GLM-4.7-Flash Q4** (agent coding on RTX 4090) |
| Desktop 48+ GB (dense 70B) | Llama 3.3 70B (MMLU 86.0, EU OK), InternVL3-78B (vision) |
| Server single-GPU (80 GB) | GPT-OSS-120B |
| Server multi-GPU | Step-3.5-Flash, Nemotron 3 Super, Qwen3.5-122B, Ling-2.6-flash 104B |
| Workstation 256 GB (extended tier) | DeepSeek-V4-Flash 284B/A13B (native 1M ctx, FP4+FP8) |
| Long context (> 256K) | Nemotron 3 Nano (1M, RULER 86.3%), DeepSeek-V4-Flash (1M native) |
| Token-efficient agent loops | Ling-2.6-flash (15M tokens on full AA suite) |
| On-device + thinking mode | ZAYA1-8B (760M active, 8.4B total) |
| Multimodal any-to-any | Nemotron 3 Nano Omni 30B-A3B (text+audio+image+video) |
| OCR / long-doc compression | DeepSeek-OCR-2 (Apache 2.0, *Optical Compression*) |
| Linear-attention research baseline | Kimi-Linear 48B/A3B (1M ctx, KDA + MLA hybrid) |
| Extreme quantization (mobile) | Bonsai-8B (1-bit, 1.15 GB) |
| Vision on edge (< 1 GB) | LFM2.5-VL-450M |
| RL self-play research (Lean) | DeepSeek-Prover-V2-7B + SGS (Stanford) |
| Math | Nemotron Nano 9B v2 (/think mode), GPT-OSS-120B |
| Code (real bugs) | Step-3.5-Flash, Devstral Small 2 |
| Code (competition) | GPT-OSS-120B (Codeforces 2622) |
| Multilingual (100+ langs) | Qwen 3.5 (201), Qwen 3 (119) |
| Theorem proving (Lean 4) | BFS-Prover-V2-32B (95% miniF2F) |
| Theorem proving (natural language) | Nomos 1 (Putnam 2025: 87/120 with harness) |
| RAG with literal citations | Pleias-RAG-1B (native citation, 1B) |
| Copyright-safe training data | Pleias (100% public domain / CC) |
| Agent coding (laptop) | OmniCoder-9B (Terminal-Bench 23.6%), GLM-4.7-Flash Q4 |
| Competitive coding (LCB) | NousCoder-14B (LCB v6 67.87%) |
| Uncensored generalist | Hermes 4-70B (SOTA RefusalBench) |
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
| Kimi K2.5 | 🔴 Modified MIT (100M MAU) | Listed-with-warning candidate; left out pending updated fiche — see warning callout in [Generalists](#generalists) for the principle |
| DeepSeek V3/R1 full (671B) | MIT | Q4 ≈ 370 GB, beyond 256 GB extended cap |
| DeepSeek-V4-Pro (1.6T/49B) | MIT | Q4 ≈ 800 GB, datacenter only |
| Qwen 3 235B / Qwen 3.5 397B | Apache 2.0 | 235B Q4 ≈ 130 GB and 397B Q4 ≈ 218 GB technically fit extended tier — left out as MoE generalist territory is already covered by Ling-2.6-flash 104B and DeepSeek-V4-Flash with better efficiency |

---

## Contributing

Found an error? Missing a model? Open an issue or submit a PR.

Sources: [HuggingFace](https://huggingface.co), [Papers With Code](https://paperswithcode.com), official model repos and papers.

---

## License

This list is licensed under [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).

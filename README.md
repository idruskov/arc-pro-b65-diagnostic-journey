# Intel Arc Pro B65 — diagnostic journey (2026-11-04 to 2026-11-06)

## TL;DR

Short version: a brand-new Intel GPU gave us gibberish instead of text, and it took two days to find out why — twice, in two unrelated parts of the stack.

A local-inference investigation on Intel Arc Pro B65 found correctness failures in two separate parts of the Intel GPU software stack: an FP16/oneDNN path and a Vulkan long-context path. Neither is the same bug, and we were careful not to assume they were just because they both lived on the same card. Controlled workarounds made ordinary inference usable, and same-device speculative decoding pushed an earlier Qwen baseline from 15.83 tok/s to roughly 25.9–26.1 tok/s — a real gain, not a rounding error.

Then a longer, RAG-style request broke it again, in a new way. Repeated `333` output, no coherent answer, reproducible with the draft model removed and with thinking mode on or off. That one is still open, tracked in [llama.cpp #31828](https://github.com/ggml-org/llama.cpp/issues/31828).

## Hardware and software

- Primary inference GPU: Intel Arc Pro B65, 32 GB
- Secondary GPU: NVIDIA RTX 3050, 6 GB
- Platform: virtualized Ubuntu inference guest with PCI passthrough
- Primary runtime: llama.cpp with Vulkan
- Plan Omega target: Qwen3.8-27B Q5_K_M GGUF
- Speculative draft: Qwen3.8-27B DFlash2 Q4_K_M GGUF
- Embeddings: BGE-M3 Q4_K_M on RTX 3050 CUDA
- Mesa versions compared: 26.1.6 and 26.2.6
- Current llama.cpp test build: 86a283532072722c5f3363d37d59a874d09fa99b

This document omits credentials, addresses, hostnames, private repositories, model-download sources, and local filesystem paths.

## Timeline

### Day 1: discovering the failure mode

Initial GPU-backed tests occasionally returned obvious non-language output — repeated punctuation, repeated characters, nothing resembling an answer. The easy move here is to blame the prompt, or the model, or call it noise and rerun. We didn't do that. A GPU that occasionally returns `!!!!` instead of a sentence is not a benchmarking quirk; it's a correctness bug until proven otherwise.

That became the rule for the rest of the investigation: a passing health check proves the service is up, not that it's right. A short logic test proves that one prompt, not the model. Fast is not the same as correct. We kept re-learning this the hard way, which is partly why this document exists.

### oneDNN FP16 bug

A separate FP16 correctness problem was isolated on an Intel GPU software path:

- FP16 matrix results could be non-finite or disagree with controls while FP32 controls passed.
- Passing GPU microkernels were not treated as proof for all matrix-engine paths.
- Isolation experiments implicated the enabled oneDNN path in the affected setup.
- A prior FP32 fix was not treated as proof for the FP16 failure.
- Hardware and platform causes were not claimed fully excluded.

Details and reproducer discussion are in [oneDNN issue #6440](https://github.com/uxlfoundation/oneDNN/issues/6440). PyTorch/XPU and vLLM/XPU observations were kept separate from the Vulkan investigation.

### Vulkan cooperative-matrix workaround

For llama.cpp Vulkan, the following environment workarounds improved correctness in the earlier affected scenario:

~~~bash
export GGML_VK_DISABLE_COOPMAT=1
export GGML_VK_DISABLE_COOPMAT2=1
export GGML_VK_FORCE_MMVQ=1
~~~

The workaround was compared across Mesa 26.1.6 and Mesa 26.2.6. It narrowed the observed trigger conditions, but did not prove correctness for every Vulkan workload.

### Hardware exclusion tests

Bounded tests included repeated large VRAM allocation and access checks, direct DPAS/XMX-oriented controls, deterministic generation, and clock/throttle observations.

These tests did not show a clear hardware fault in exercised paths. They do not prove all hardware or firmware causes impossible; they shifted priority toward reproducible software-path investigation.

### Plan Omega — dual-GPU speculative decoding

Plan Omega aimed to accelerate generation with DFlash2 on RTX 3050 CUDA while the Qwen target ran on B65 Vulkan.

The cross-vendor path encountered practical constraints:

- A speculative scheduler/shared-tensor issue required the upstream work described in [llama.cpp PR #26636](https://github.com/ggml-org/llama.cpp/pull/26636).
- Mixed-vendor target/draft execution is technically possible but weakly supported relative to a single-device path.
- Large-context speculative decoding can be harmed by exposing multiple Vulkan devices; [llama.cpp issue #30695](https://github.com/ggml-org/llama.cpp/issues/30695) provides relevant upstream context.
- DFlash has tensor-split constraints; see [llama.cpp issue #27833](https://github.com/ggml-org/llama.cpp/issues/27833).

The usable result turned out to be the boring one: skip the second card entirely. Target and DFlash2 both on B65, with RTX hidden from Vulkan enumeration so it can't quietly interfere:

~~~bash
export GGML_VK_VISIBLE_DEVICES=0
llama-server -m Qwen3.8-27B.Q5_K_M.gguf -md Qwen3.8-27B-DFlash2-Q4_K_M.gguf --spec-type draft-dflash --device Vulkan0 --device-draft Vulkan0 -ngl 99 -ngld all --spec-draft-n-max 7 -c 32768 -np 1 -fa on -ub 256 -ctkd q8_0 -ctvd q8_0 --no-context-shift -fit off
~~~

The RTX 3050 was reassigned to an independent CUDA embeddings service instead, where it earns its keep without sitting anywhere near Qwen's decode path.

### An unrelated scare: a four-minute warm-up

While recovering from one of the speculative-decoding experiments, the server came back up but sat at HTTP 503 for several minutes during llama.cpp's Vulkan warm-up. A read-only stack inspection pointed at `xe_validation_lock` inside the kernel — the kind of thing that makes you assume the worst, specifically a hung GPU driver or leaked device state from a day of starting and stopping Vulkan contexts.

It wasn't that, or at least nothing proved it was. The kernel log for that window showed nothing — no Xe/DRM timeout, no GPU reset, no hang record, just ordinary initialization. Warm-up simply finished on its own after about four minutes, and a follow-up request answered correctly with no warning in the log. We added `--no-warmup` as a practical, reversible fix (startup now reaches healthy in about twelve seconds instead of four minutes), and left the kernel question exactly where the evidence leaves it: a long wait that looked alarming and wasn't explained, not a confirmed driver defect.

### RAG regression: open problem

Just when the dual-GPU story looked finished, it reopened. A local RAG flow embedded a handful of reviewed documents on RTX 3050, retrieved relevant context, and fed it into a Qwen Jinja chat request on B65 — the kind of request this whole setup exists to serve.

A long, structured prompt with retrieved context sometimes degenerated into the same `333` repetition from day one. Same symptom, different trigger, months (well, days) after we thought it was handled.

| Test | Result |
|---|---|
| Simple deterministic 1 through 300 control with Q5 + DFlash2 | PASS 3/3 |
| Long RAG/Jinja prompt with DFlash2 and thinking disabled | FAIL: repeated 333 output |
| Same long RAG/Jinja prompt without DFlash2 | FAIL: repeated 333 output |
| Same long RAG/Jinja prompt with default thinking | FAIL: repeated 333 output in reasoning trace |

DFlash and thinking mode are therefore not sufficient explanations. The leading scope is a base Q5 Vulkan path or long Jinja-context interaction; causal attribution remains open.

Technical detail and a synthetic reproduction shape are in [llama.cpp #31828](https://github.com/ggml-org/llama.cpp/issues/31828).

## Measured results

| Configuration | Generation result | Interpretation |
|---|---:|---|
| Vulkan Qwen baseline | 15.83 tok/s | Earlier production smoke baseline |
| B65 target + RTX 3050 CUDA DFlash2 draft | approximately 18.5 tok/s | Works, but cross-vendor overhead limited benefit |
| B65 target + B65 Vulkan DFlash2 draft | approximately 25.9–26.1 tok/s | Earlier validated same-device result; about +64% over 15.83 tok/s |
| Same-device full 1 through 300 validation | PASS 3/3 | Exact sequence and 98.76% draft acceptance |
| Post-recovery numeric control | 46.06–46.18 tok/s | Repeated pass, but unexplained versus earlier data; not an established capacity baseline |

## Current recommended configuration

For routine local use:

- Qwen target plus DFlash2 draft on one B65 Vulkan device;
- cooperative-matrix workarounds enabled;
- RTX 3050 used independently for embeddings/RAG rather than as draft GPU;
- correctness checked with more than short smoke prompts.

The RAG/long-context regression is unresolved. Do not treat health checks or short deterministic prompts as sufficient validation for complex Jinja/RAG workloads.

## Methodology and acknowledgements

None of this required special tools. It required refusing to skip a few boring steps:

- state what was actually observed, separately from what we think explains it;
- keep flags, model identities, prompts, seeds, and timing metadata, not just the headline number;
- change one variable at a time, even when that's slower;
- when a test fails, keep the failure — don't quietly swap in a cleaner benchmark and move on;
- a `200 OK` means the service answered, not that the answer was right.

That discipline is the actual deliverable here, more than any single tok/s figure. It's why we caught the RAG regression instead of shipping a config we'd have had to walk back later. Thanks to the llama.cpp, Mesa, Intel graphics, and oneDNN communities — their public issues and discussions are most of the reason this testing had anywhere to start from.


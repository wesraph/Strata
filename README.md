<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="apps/host/public/brand/strata-lockup-dark.svg">
    <img src="apps/host/public/brand/strata-lockup-light.svg" alt="Strata" width="340">
  </picture>
</p>

<h3 align="center">Run powerful AI on your own computer.</h3>
<p align="center">Chat, code and work with images, locally.</p>

<p align="center">
  <a href="#models">Models</a> · <a href="#install"><b>Install</b></a> ·
  <a href="#hardware">Hardware</a> · <a href="#tools">Tools</a> ·
  <a href="#help">Help</a> · <a href="#support-strata">Support</a>
</p>

<p align="center">
  <a href="https://github.com/Niko1221/Strata/releases/latest"><img src="https://img.shields.io/github/v/release/Niko1221/Strata?label=download&color=0ab887" alt="Download Strata"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/Niko1221/Strata" alt="MIT license"></a>
  <a href="https://buymeacoffee.com/strataengine"><img src="https://img.shields.io/badge/Buy_me_a_coffee-FFDD00?logo=buymeacoffee&logoColor=000" alt="Support Strata on Buy Me a Coffee"></a>
</p>

<p align="center"><sub><b>English</b> · <a href="README.zh-CN.md">简体中文</a> · <a href="README.ja.md">日本語</a> · <a href="README.de.md">Deutsch</a> · <a href="README.fr.md">Français</a> · <a href="README.es.md">Español</a> · <a href="README.pt-BR.md">Português</a> · <a href="README.pl.md">Polski</a></sub></p>

> **This is a fork of [Niko1221/Strata](https://github.com/Niko1221/Strata).** It tracks upstream (currently v0.2.7) and adds a
> shared, paged K/V memory model for the engine. The section below says why and what it changes; everything after it is
> the upstream README.

## About this fork: a shared K/V pool for agents

We run coding agents against Strata: several conversations at once, each 80K to 200K tokens long, on two RTX 3090s with a
MoE model (Qwen3.8-Flash-Next, IQ4_XS) whose experts live partly in VRAM and partly in RAM. Upstream's memory model gives
every session (the main one plus each batch slot) its own contiguous K/V buffer sized up front. For that workload it meant:
the context had to be capped below the model's 262K, a slot switch copied the whole K/V image (about 4.5 s for 25K tokens),
the agents' shared system prompt was stored once per session that held it, prefix reuse only happened at checkpoints on a
token schedule, and the VRAM carved out for K/V sat idle while the expert cache, which decides decode speed, went without.

The fork replaces that with one flat pool of 4-token pages per card (`--kv-share`, branch `feature/kv-share`, merged on
`main`). The comparison, measured on the rig above:

| | upstream model | this fork |
|---|---|---|
| K/V layout | one contiguous buffer per session, sized at startup | one paged pool per card, shared by every session |
| context per conversation | capped by the per-session budget | the model's full 262K (the pool holds 684K cells per card) |
| growth during an answer | the whole buffer reserved | pages added on demand with 2K tokens of headroom |
| shared prefix (system prompt) | once per session holding it | once; pages are keyed by a hash chain over their tokens and mapped by every conversation that shares them |
| prefix reuse across turns | at checkpoints (`--prompt-cache-every`) | any prefix at page granularity: a turn reads only its new tokens (83,449 of 83,888 reused on a typical agent turn) |
| slot switch | copy the K/V image: ~4.5 s per 25K tokens | copy the page table: 30 ms for 83K tokens |
| a conversation that leaves its slot | host image, full restore on return | pages stay in the pool while there is room, else evicted least-recently-used to a host swap sized from free RAM; parks skip every pooled page |
| VRAM for K/V | fixed carve-out, taken from the expert cache whether used or not | borrowed from the expert caches on demand and returned after a minute idle (the VRAM exchange), so experts get the VRAM back when conversations rest |

Measured against a real build of upstream 0.2.7 on the rig above, same config minus the pool, with an agent loop:
each agent has the tool-definition system prompt and a 40K-token repository context, works turn by turn with its
tool calls executed (read-only shell and file reads, writes into a workspace), and stops after 6K generated tokens;
contexts grow to 90K-130K tokens. "Aggregate" is tokens generated over the agent's whole wall clock, prompt reads
and tool waits included - what an agent experiences.

| agents | | upstream 0.2.7 | this fork (two runs) |
|---|---|---|---|
| 1 | aggregate over the agent loop | 38 tok/s | 56-69 tok/s |
| | decode on the long answers | 86 tok/s | 107-126 tok/s |
| | wait before the first token, later turns | 1.7 s | 1.4-1.6 s |
| 2 | aggregate | 43 tok/s | 54-66 tok/s |
| | decode on the long answers, per agent | 31 and 51 tok/s | 49-66 and 66-80 tok/s |
| | wait before the first token, later turns | 2.2-2.8 s | 1.5-2.7 s |
| 3 | aggregate | 56 tok/s | 57 tok/s |
| | decode on the long answers, per agent | 24-40 tok/s | 30-53 tok/s |
| | expert cache hit rate at the end | 35-63% | 96-99% |

The agents choose their own tool calls (temperature 0.6), so two runs of the same arm do different work: the
aggregate moves with how many long modules the run happened to write, the decode and the hit rate do not.

Two pipelined proposals per slot (`STRATA_PIPED_MTP=2`) pay on this traffic: three of four proposals are accepted
when agents write code, and the same build measured 85 tok/s aggregate at two agents and 72 tok/s at three (67-83
and 30-67 tok/s on the long answers), unchanged at one. It is the setting the fork's own config runs. Three
proposals (the most two slots fit in a window) lose again: 73 and 64 tok/s at two and three agents, with 69-77% kept.

Prefill is unchanged (about 3.6K tok/s for 46K to 237K-token first reads). The three-agent aggregate is even
because three different coding tasks keep the engine's prompt path and expert reads busy either way; the pool's
gain there is the hit rate: upstream's model carves four 262K sessions out of VRAM up front and keeps 3,805 expert
slots, the pool leaves 5,648.

**The same load, replayed.** The agent loop above does different work on every run, so the fork also replays a
recording of it: 80 recorded turns of three agents (29,148 generated tokens, every answer run to its recorded
length, temperature 0), the same bytes for both engines, free-running as the agents were. Measured 2026-10-29 on
the rig above against a real upstream 0.2.7 build, same config minus the pool:

| | upstream 0.2.7 | this fork |
|---|---|---|
| wall clock for the three agents to finish | 584 s | 368 s |
| aggregate (29,148 tokens over that wall) | 49.9 tok/s | 79.2 tok/s |
| wait before the first token, later turns, median / p90 | 2.73 s / 5.11 s | 1.55 s / 4.43 s |
| decode on the long answers, per agent | 31.8 / 23.7 / 27.3 tok/s | 55.2 / 35.4 / 48.0 tok/s |
| reading a turn's new tokens, under 300 / 300-1,000 / 1,000-2,500 | 914 / 1,952 / 3,034 ms | 418 / 1,021 / 1,757 ms |
| reading the 41K-token first prompt | 13.8 s | 12.0 s |

The same replay on the UD-Q4_K_XL pack (104 GB; experts 20% bigger, so each card's expert cache holds a third
fewer of them and every read and miss leans harder on the x8 links), same day, one run per arm:

| | upstream 0.2.7, Q4_K_XL | this fork, Q4_K_XL |
|---|---|---|
| wall clock for the three agents to finish | 852 s | 588 s |
| aggregate (29,148 tokens over that wall) | 34.2 tok/s | 49.6 tok/s |
| wait before the first token, later turns, median / p90 | 4.20 s / 8.69 s | 2.60 s / 7.28 s |
| decode on the long answers, per agent | 22.0 / 15.1 / 19.5 tok/s | 33.7 / 23.1 / 28.2 tok/s |
| reading a turn's new tokens, under 300 / 300-1,000 / 1,000-2,500 | 1,894 / 3,030 / 4,402 ms | 1,237 / 2,115 / 3,099 ms |
| reading the 41K-token first prompt | 19.3 s | 14.4 s |
| expert cache per card | 3,183 slots | 4,790 slots |

The pool leaves the caches 1,600 more slots per card because no K/V is carved out up front, and the CPU share's
measured cap landed at 2,048 tokens on this pack (4,096 on IQ4_XS): the 2K-4K band read 16% cheaper streamed here.

Three things closed the gap on the reads since the table before it. A slot that gave its conversation back
through its turn checkpoint saved the whole K/V through host memory (3 to 5 s per turn, 26 to 36 turns per replay)
for a restore that skipped the pooled pages; the save skips them too now. The RAM copy of the experts is sized by
what the caches do not hold at the start, and with the pool they hold more, so the experts the pool later takes
had no copy and came through the page cache (2 to 3 GB per prompt); the copy keeps them now under
`--resident-budget-gib`. And the CPU share of a short prompt's streamed experts (`STRATA_PREFILL_CPU_SHARE`,
upstream's own feature) was off beside batch slots whatever the setting; on this fork it is `auto` there by
default, since the slots wait for the read anyway. The share is what the replay above runs: it moves part of the
arithmetic to the CPU kernels the engine already uses for cache misses, rounding noise at a mean KL of 0.004
against the share off (upstream's measurement; in our probes the greedy text was byte-identical once and differed
in one answer of three another time), and `STRATA_PREFILL_CPU_SHARE=0` turns it off.

**Using it.** `--kv-share auto` has no fixed share: the pool may use the free VRAM of the tightest card, starts at
one extent and grows with the sessions, borrowing from the expert caches and handing back what the last
`STRATA_KV_TRIM_WINDOW_S` seconds (600; we run 30) did not touch (`STRATA_KV_SHARE_FRAC=F` keeps a fixed split); the host
swap (`--kv-swap-mib`) sizes itself from free RAM when unset; the exchange is on by default (`STRATA_KV_EXCHANGE=0` turns
it off, `STRATA_KV_TRIM_IDLE_S` and `STRATA_KV_TRIM_KEEP_TOKENS` tune the idle trim). `STRATA_STATS_EVERY_S=1` prints a
line per second with the prompt and generation rates over the interval, the slots running and reading, the hit rate
and the pool's use. The flags are documented in [docs/DETAILS.md](docs/DETAILS.md). With two NVIDIA cards,
`STRATA_PIPED_MTP=2` and `--batch-mtp` add the pipelined slot drafts that the tables above use. The CPU share of
the short reads is on by default beside the slots on this fork, its fraction measured per layer and the chunk size it
applies up to measured per size band (`STRATA_PREFILL_CPU_SHARE=0` turns it off). Give the engine headroom: the pool shares the card's free VRAM with everything else, and
`--vram-reserve-mib 1536` is what keeps three or more long contexts from paging on 24 GB cards.

**Known edges.** Every mapped page is pinned, so three or more conversations near 200K tokens can fill the pool and the next
answer waits on evictions. The exchange's working-set window and minimum lend are tuned on our traffic, not derived.
Unpinned mappings for idle slots and a fairer share rule near a full pool are the next steps.

---

**Free, open source, and yours to run.** Use Strata on your desktop or with your favorite coding tools. No account needed.

## Models

We recommend **ISTA-DASLab first**, then **Unsloth** and other providers, where a compatible model fits.

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.8-Flash-Next</b></summary>

| Provider | Choices |
| --- | --- |
| **ISTA-DASLab · recommended** | Q2\_0, IQ2\_XS, IQ3\_XXS, IQ3\_S. [Coder: IQ1\_M](docs/MODELS.md#coder) for coding. |
| **Unsloth** | UD-IQ4\_XS, UD-Q4\_K\_XL, UD-Q5\_K\_XL, UD-Q6\_K\_XL, Q8\_0 |
| **Gyro · agentionai** | [Gyro-S, Gyro-M](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF). Lower memory use but slower. |
| **Other options** | [Swift 1.5](docs/MODELS.md#swift-15), [NVFP4](docs/NVFP4.md), [OrcaRouter](docs/ORCA.md) |

[Compare sizes, memory needs and compatibility](docs/MODELS.md).

</details>

<details>
<summary><img src="docs/media/models/qwen.svg" width="22" alt="" align="middle"> &nbsp;<b>Qwen3.6-35B-A3B</b></summary>

**Unsloth:** UD-Q4_K_XL, UD-Q6_K_XL, Q8_0. These are the currently documented packs for this model.

[Model guide and compatibility](docs/QWEN36.md).

</details>

<details>
<summary><img src="docs/media/models/deepseek.svg" width="22" alt="" align="middle"> &nbsp;<b>DeepSeekV4.1-Flash-1017</b></summary>

In the **desktop app** and the **Terminal**. What does not fit in memory is read from your SSD, so it needs a lot of free
disk space.

| Set (IST-DAS) | Download | Free disk needed |
| --- | ---: | ---: |
| 1.75 bpw | 239 GB | about 245 GB |
| 2.00 bpw | 257 GB | about 265 GB |
| 2.25 bpw | 274 GB | about 280 GB |

Runs on NVIDIA RTX 30, 40 and 50 series (Windows and Linux), AMD Radeon AI PRO R9700 (Linux), and AMD Ryzen AI Max — Strix Halo
(Windows). Setup downloads the files and the DeepSeek engine, then starts the model.

```sh
strata launch --family deepseek41 --model ist-das-2.00bpw --plan
```

Measured with the 2.00 bpw set on Strata 0.2.5 (Strix Halo: 0.2.7). Output is the speed of the answer, prompt is how fast Strata reads your input.

| Hardware | Output | Prompt |
| --- | ---: | ---: |
| RTX 5070 12 GB, 64 GB RAM | 4 to 6 tokens/s | about 110 tokens/s (4K) |
| RTX 3060 12 GB, 46 GB RAM | 9 to 11 tokens/s | about 155 tokens/s (4K) |
| 1x Radeon AI PRO R9700, 251 GB RAM | 21 to 25 tokens/s | 320 to 360 tokens/s (8K) |
| 4x Radeon AI PRO R9700 | 74 to 77 tokens/s | 400 to 450 tokens/s (8K) |
| Ryzen AI Max+ 395 (Strix Halo), 128 GB unified | 19 to 21 tokens/s (code), 11 (chat) | 87 tokens/s (4K), 125 tokens/s (8K) |

[Memory, disk and details](docs/DEEPSEEK41.md) · [Release notes](https://github.com/Niko1221/Strata/releases)

</details>

<a id="the-other-sizes"></a>

Not sure what fits? [Choose a model for your computer](docs/MODELS.md#pick-by-memory).

## Install

<a id="option-a-the-strata-app"></a>

### Desktop app

**[Download Strata for Windows, Linux or Mac →](https://github.com/Niko1221/Strata/releases/latest)**

Install the app, choose a model, and start chatting. There is one installer for each system. On first start the app
downloads the engine for your graphics card. [Help with the installer](docs/INSTALL.md#the-app).

<sub>Files: `Strata-0.2.7-windows-x64-setup.exe` · `Strata-0.2.7-linux-x86_64.AppImage` · `Strata-0.2.7-macos-arm64.dmg`</sub>

<p align="center">
  <img src="docs/media/app-chat.png" width="900" alt="Strata desktop app with a chat and live code preview side by side">
  <br><sub>Chat, files and a live preview in one window.</sub>
</p>

<a id="option-b-the-terminal-one-line"></a>

### Terminal app

Prefer the terminal? Install Strata, then run **`strata launch`**.

<details open>
<summary><b>Install commands</b></summary>

**Linux / Mac**

```sh
curl -fsSL https://github.com/Niko1221/Strata/releases/latest/download/install.sh -o install.sh && sh install.sh
```

**Windows PowerShell**

```powershell
irm https://github.com/Niko1221/Strata/releases/latest/download/install.ps1 | iex
```

On Windows, open a new terminal afterwards. Then:

```sh
strata launch
```

Follow the prompts to download and start a model. [Terminal guide](docs/TERMINAL_LAUNCH.md).

</details>

<p align="center">
  <img src="docs/media/terminal-monitor.png" width="900" alt="Strata terminal app showing model activity, speed, memory and recent requests">
  <br><sub>Watch your model and hardware while it runs.</sub>
</p>

<a id="build-from-source"></a>

[Build from source](docs/SOURCE_INSTALL.md) · [Docker](docs/DOCKER.md) · [Full install guide](docs/INSTALL.md)

<a id="what-you-need"></a>

## Hardware

| Supported hardware | Guide |
| --- | --- |
| **NVIDIA GeForce RTX 20 / 30 / 40 / 50 series** | [NVIDIA setup](docs/INSTALL.md#what-you-need) |
| **AMD Radeon RX 7700 / 7800 / 7900, RX 9060 / 9070, AI PRO R9700** | [AMD setup](docs/AMD_HIP.md) |
| **AMD Radeon RX 6800 / 6900 · experimental** | [RX 6000 setup](docs/AMD_HIP.md#rdna2-gfx1030) |
| **Intel Arc · Linux** | [Intel setup](docs/INTEL_ARC.md) |
| **Apple Silicon · M1 or newer** | [Mac setup](docs/MAC.md) |
| **AMD Ryzen AI Max · Strix Halo** | [Strix Halo setup](docs/STRIX_HALO.md) |

<a id="using-it"></a>

## Tools

**Claude Code · Codex CLI · OpenCode · oh-my-pi · pi.dev · Hermes Agent · OpenClaw**, and more through the OpenAI- and Anthropic-compatible APIs.

Run **`strata connect`** to see setup instructions for your tool. [Connection guide](docs/TERMINAL_LAUNCH.md#connect-your-apps-strata-connect).

<a id="guides"></a>
<a id="how-fast-is-it"></a>
<a id="how-does-it-work"></a>
<a id="something-went-wrong"></a>

## Help

[All guides](docs) · [Troubleshooting](docs/TROUBLESHOOTING.md) · [Report a bug](https://github.com/Niko1221/Strata/issues/new/choose) ·
[Benchmarks](docs/COMMUNITY_BENCHMARKS.md) · [How it works](docs/HOW_IT_WORKS.md) · [Contribute](CONTRIBUTING.md)

## Support Strata

**Help keep local AI open to everyone.** A coffee helps fund development and testing on more hardware.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me a Coffee: support Strata" height="48"></a>

[MIT license](LICENSE) · [Credits and model licenses](docs/HOW_IT_WORKS.md#credits) · [Privacy](docs/SAFETY.md)

<a id="option-c-let-your-ai-set-it-up"></a>

<sub>For AI tools and automated readers: <a href="AGENTS.md">technical reference</a> · <a href="docs/AI_SETUP.md">setup instructions</a>.</sub>

**Sponsorships, hardware partnerships or collaborations:** [niko.veit@outlook.com](mailto:niko.veit@outlook.com).


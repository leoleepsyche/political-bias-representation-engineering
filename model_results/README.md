# Archived Model Results

This directory keeps a stable copy of the multi-model experiment outputs that were originally produced in isolated workspaces.

Each model subdirectory contains:
- `results_step1/` political layer localization
- `results_step2/` L/R/N bias analysis
- `results_step3/` topic-level analysis
- `results_step4/` steering and capability evaluation

All runs below use the current 20-topic English political dataset and the same four-step pipeline.

## Summary

| Model | Political layers | Max gap | Neutral projection | Top topic | Content/style | Capability at `alpha=0.5` |
|---|---:|---:|---:|---|---:|---:|
| `Qwen2.5-1.5B-Instruct` | `[1, 28]` | `7.37°` | `+0.5696` | `gun_control (9.21°)` | `0.63x` | `1.0` |
| `Qwen2.5-7B-Instruct` | `[1, 28]` | `15.70°` | `+2.5662` | `gun_control (17.75°)` | `0.71x` | `0.6` |
| `Qwen2.5-7B` | `[1, 28]` | `11.26°` | `+3.3359` | `gun_control (12.65°)` | `0.68x` | `1.0` |
| `Llama2-Chinese-7B-Chat` | `[1, 32]` | `14.69°` | `+0.3679` | `minimum_wage (23.55°)` | `0.73x` | `0.0` |
| `ChatGLM3-6B` | `[1, 28]` | `15.33°` | `+4.0344` | `minimum_wage (27.21°)` | `0.75x` | `0.0` |
| `Mistral-7B-Instruct-v0.2` | `[1, 32]` | `18.07°` | `+2.3741` | `minimum_wage (29.60°)` | `0.77x` | `0.0` |
| `Mistral-7B-v0.1` | `[1, 32]` | `13.47°` | `+0.8904` | `gun_control (20.80°)` | `0.67x` | `1.0` |

## Interpretation Notes

- Across all archived runs, the detected political range spans almost the full transformer stack (`[1, 28]` or `[1, 32]`), so these should be read as politically discriminative layer ranges rather than exclusive political modules.
- Every model still shows substantial `style` entanglement. The content/style ratio stays below `1.0x` in all runs.
- `Instruct` variants tend to show stronger political separation than their base counterparts, but the steering tradeoff differs by model.

## Directory Map

- `qwen2.5-1.5b-instruct/`
- `qwen2.5-7b-instruct/`
- `qwen2.5-7b/`
- `llama2-chinese-7b-chat/`
- `chatglm3-6b/`
- `mistral-7b-instruct-v0.2/`
- `mistral-7b-v0.1/`

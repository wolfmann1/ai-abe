# Choosing a local model for your graphics card

A local model runs on your graphics card (GPU) through Ollama. It has to fit in the card's own memory, called
VRAM. If it doesn't fit, it either fails to load or runs partly on the main processor and becomes very slow.
This page shows how to pick one that fits.

## 1. Find out how much VRAM you have

On Windows, open **Task Manager → Performance → GPU** and read **Dedicated GPU memory**. With an NVIDIA card you
can also run `nvidia-smi` in a terminal.

Check the number rather than going by the card's name: the same model name can come with different amounts of
memory, especially between laptops and desktops.

## 2. Use the fitting rule

> **Download size + about 2 GB ≤ your VRAM**

The download size is listed on each model's **Tags** page on [ollama.com/library](https://ollama.com/library).
The extra 2 GB holds the text the model is working on (your question and the document passages ABE sends with
it) and leaves room for Windows to draw the screen.

## 3. Pick a model

Starting points by card size, using Google's Gemma 4 sizes from
[Ollama's tag list](https://ollama.com/library/gemma4/tags) and Meta's Llama 3.1:

| Your VRAM | Largest download that fits comfortably | Good choices |
|---|---|---|
| 6 GB | about 4 GB | `gemma4:e2b-it-qat` (4.3 GB) |
| 8 GB | about 6 GB | `gemma4:e4b-it-qat` (6.1 GB), `llama3.1:8b` (4.9 GB) |
| 12 GB | about 10 GB | `gemma4:12b-it-qat` (7.2 GB) |
| 16 GB | about 14 GB | `gemma4:12b-it-q8_0` (13 GB) |
| 24 GB | about 22 GB | `gemma4:26b-a4b-it-qat` (16 GB), `gemma4:31b-it-qat` (19 GB) |

Bigger models usually answer better and always answer slower. The way to know whether a bigger one is worth it
for your documents is to run `abe eval` with each and compare the reports.

## What the tag names mean

| Part of the tag | Meaning |
|---|---|
| `e2b`, `e4b`, `12b`, `26b`, `31b` | Model size in billions of parameters. More parameters means more knowledge and more memory. The `e` means "effective": these smaller Gemma models are built for laptops and phones. |
| `a4b` | A "mixture of experts" model: only about 4 billion of its parameters work on each word, so it runs faster than its size suggests, but the whole model still has to fit in memory. |
| `it` | Instruction-tuned: trained to follow instructions and answer questions. Use these. |
| `qat` | Quantization-aware training. The model was trained to run in compressed form, so it keeps more of its quality than the standard compressed versions. Prefer it when it's offered. |
| `q4_K_M`, `q8_0` | Compression level. `q4` stores each number in about 4 bits and is the usual choice; `q8` is roughly twice the size and slightly more accurate. |
| `bf16` | Uncompressed. About four times the size of `q4`; rarely worth it on a home card. |
| no tag, or `latest` | Ollama's default for that model, usually a mid-size `q4`. Check its size before assuming it fits. |

## 4. Install it and point your agent at it

```
ollama pull gemma4:e4b-it-qat
```

Then either choose it in the form when you build the agent, or change `model:` in the agent's `agent.yaml`:

```yaml
provider:
  kind: openai_compatible
  endpoint: http://localhost:11434/v1
  model: gemma4:e4b-it-qat
```

No rebuild is needed after changing the model. The document index doesn't depend on it.

## 5. Check it's running on the GPU

Ask the agent a question, then run:

```
ollama ps
```

The **PROCESSOR** column should read **100% GPU**. A split such as `40%/60% CPU/GPU` means part of the model
didn't fit and answers will be several times slower. Pick a smaller model or free some VRAM.

## When it doesn't fit

The error looks like this in ABE:

```
openai_compatible returned HTTP 500: ... llama-server reported out-of-memory during startup ...
CUDA error: out of memory
```

In order of what to try:

1. **Free VRAM.** `ollama ps` lists loaded models; `ollama stop <model>` unloads one. Close games, video
   editors and other programs that use the GPU.
2. **Use a smaller tag** of the same model, such as `e4b` instead of `26b`, or `qat` instead of `q8_0`.
3. **Restart Ollama** if errors continue after freeing memory. A crashed load can leave memory held until it
   restarts.

The first answer after loading a model takes longer, often 10 to 30 seconds, because Ollama is moving the model
into VRAM. Later answers are faster while it stays loaded.

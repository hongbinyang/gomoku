# Compute devices

Training and play accept:

```text
--device auto|cpu|cuda|mps|tpu
```

`auto` selects TPU when the runtime declares `PJRT_DEVICE=TPU`; otherwise it
prefers CUDA, then Apple MPS, then CPU.

| Value | Backend | Requirements |
| --- | --- | --- |
| `auto` | Best detected backend | No additional configuration |
| `cpu` | Host processor | Standard PyTorch installation |
| `cuda` | NVIDIA GPU | CUDA-enabled PyTorch and driver |
| `mps` | Apple GPU through Metal | Apple Silicon, supported macOS, MPS-enabled PyTorch |
| `tpu` | TPU through PyTorch/XLA | TPU runtime and compatible `torch_xla` installation |

Examples:

```bash
python -m gomoku_muzero.train --device mps
python -m gomoku_muzero.train --device cuda
python -m gomoku_muzero.train --device tpu
python -m gomoku_muzero.play --device auto
```

Check Apple MPS locally:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

Check CUDA:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

TPU support is optional and is not installed as a base project dependency.
Install the PyTorch/XLA build appropriate for the TPU runtime. PyTorch/XLA is
lazy, so the trainer explicitly synchronizes after optimizer steps.
The current background actor is thread-based, so TPU training must also pass
`--self-play-mode sync` until process-based XLA actors are implemented:

```bash
python -m gomoku_muzero.train --device tpu --self-play-mode sync
```

Checkpoints always store CPU tensors. A checkpoint trained on CUDA, MPS, or
TPU can therefore be loaded on a different supported backend.

## Performance expectations

The accelerator speeds up neural-network operations. Current MCTS remains a
sequential Python search and performs batch-size-one inference, so accelerator
utilization can be low during self-play. Device support makes execution
portable; independent self-play and batched inference are separate scaling
steps.

References:

- [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch CUDA](https://docs.pytorch.org/docs/stable/cuda.html)
- [PyTorch/XLA](https://docs.pytorch.org/xla/)

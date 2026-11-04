# HW4.a + HW4.5: BERT fine-tuning, vLLM serving, training throughput, and memory

SLICE = 384 and SEED = 1384, derived from SID4 = 1384. Every script writes JSON results to `results/`, tags the output with the GPU name, and records the run in `RUN_LOG.txt`.

## Environment

The experiments were performed on lab machine ADS-R15-836-04 using an NVIDIA RTX 4090 with 24 GB VRAM and Windows 11.

- **BERT training and Parts A-B:** Windows Python 3.13
- **vLLM serving, KV-cache measurements, and DDP baseline:** WSL2 Ubuntu 24.04
- **Training PyTorch:** 2.11.0+cu128
- **Serving PyTorch:** 2.8.0+cu128
- **Training Transformers:** 5.20.2
- **Serving Transformers:** 4.57.6
- **vLLM:** 0.11.0
- **bitsandbytes:** 0.50.2

Install the training packages with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets scikit-learn pandas bitsandbytes accelerate matplotlib

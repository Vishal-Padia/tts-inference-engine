# tts-inference-engine
Trying to build an TTS Inference Engine

### Reason for building this?
I am learning about CUDA and GPU Optimization. A learning by building is something I enjoy, that's why I'm building this. Also why TTS engine rather than other things like LLM Inference Engine or Image Generation Engine? Because LLM Inference Engine is something many people are building and Image Generation Engine sounds kind off difficult considering my current knowledge, so I thought TTS engine would be a good start and also not many people are building it.

### Research papers:

**FastSpeech: https://arxiv.org/pdf/1905.09263**

**FastSpeech2: https://arxiv.org/pdf/2006.04558**

Fastspeech is the moment TTS became "inference-friendly".

Read For:
- Duration predictor role
- Length regulator mechanics
- Where sequence expansion happens
- Which ops become large batched GEMMs

**HiFi-GAN: https://arxiv.org/pdf/2010.05646**

This is the real inference bottleneck in TTS. Convolution-heavy, GPU-bound, widely used

Read For:
- Exact convolution stack
- Residual block structure
- Kernel sizes and dilation patterns
- Where latency concentrates during inference

**VITS: https://arxiv.org/pdf/2106.06103**

Flow-based decoder + GAN vocoder coupling

Read For:
- Inference path only
- Which components run sequentially
- Where memory access becomes irregular
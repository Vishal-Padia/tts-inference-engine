# Fastspeech 2: Fast and High-Quality End-to-End Text to Speech

It's a non-autoregressive (parallel) TTS model that improves upon FastSpeech 1 by removing the dependency on an external teacher model (like tacotron2).

Instead of learning from a teacher's simplified output (knowledge distillation), Fastspeech 2 learns directly from the ground truth audio. To make this harder task possible (solving the one to many mapping problem without a teacher guiding it), it explicitly conditions the generation on Duration, Pitch and Energy.

# 1. The Solution: Explicit Variation Modeling

The "one to many mapping" problem means text doesn't contain enough information to generate specific speech. Fastspeech 1 cheated by using a etacher model's output as the "single truth". Fastspeech 2 solves this by saying: "If we want to train on real data, we need to give the model extra hints about the specific recording."

It introduces a Variance Adaptor block that sits between the Encoder (Text) and Decoder (Mel-Spectrogram). This block injects three critical pieces of information:
1. Duration: How long is this phoneme?
2. Pitch ($F_0): What is the fundamental frequency contour?
3. Energy: How loud/intense is this frame?

Variance Adaptor is like "adding metadata" to the text embedding. Text + "Low Pitch" + "High Energy" + "Long Duration" = Unqiue, specific speech.

# 2. The Variance Adaptor Architecture:

This is the heart of the paper. The variance adaptor consists of three predictors that run sequentially.
- Duration Predictor: Same as fastspeech 1. Predicts how many mel-frames each phoneme corresponds to.
- Pitch Predictor: Predicts the pitch contour. Since pitch varies wildly, they don't just predict a raw number. They use a continuous wavelet transform (CWT) to decompose pitch into a spectrogram, predict that, and reconstruct it. Crucial for prosody.
- Energy Predictor: Predicts the L2-norma (amplitude) of the mel-spectrogram frame. Crucial for volume and stress.

# 3. Training vs Inference Strategy (The "Look Ahead" Trick)

This is the clever part that allows it to train on Ground Truth.

### During Training (Teacher Forcing with features):

We have the target audio. We extract the actual duration, pitch, and energy from the wav file. We quantize them (bucket them) and add them to the phoneme embeddings.
- Input: Text Embeddings + Ground Truth Duration + Ground Truth Pitch + Ground Truth Energy.
- Target: Ground Truth Mel-spectrogram.

### During Inference (Prediction):

We don't have the audio. We use the internal predictors (trained to guess these values) to generate the duration, pitch and energy from the text alone.
- Input: Text Embeddings + Predicted Duration + Predicted Pitch + Predicted Energy.

# 4. FastSpeech 2s (End to End)

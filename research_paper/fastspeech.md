# FastSpeech: Fast, Robust and Controllable Text to Speech

Before this, SOTA TTS models (like TacTron2) were autoregressive. This means to generate the 2nd millisecond of audio, they had to wait for the 1st to finish. This is super slow and inefficient. If the model "forgot" where it was in the sentence, it would skip words or repea them.

That's where FastSpeech comes in. It's a non-autoregressive model that can generate audio in parallel. Fastspeech creates Mel-Spectrograms in parallel. It looks at the whole text at once and spits out the whole audio representation at once. This makes it ~270x faster Mel-Spectrogram generation than TacTron2 (claim in the paper). 

## The breakdown:

1. Parallel Mel-Spectrogram Generation
2. Phenome Duration Predictor
3. Length Regulator

These are the 3 main components of FastSpeech. 

## Parallel Mel-Spectrogram Generation:

This is the heart of the paper, the entire Mel-Spectrogram is treataed as a single tensor $M$ [Batch, Time, Mel_Channels].

#### High-level Pipeline:
The model is a feed-forward network, so there are no loops.

$$ \text{Text} \xrightarrow{\text{Encoder}} H_{phoneme} \xrightarrow{\text{Length Regulator}} H_{mel} \xrightarrow{\text{Decoder}} \text{Mel-Spectrogram} $$

#### The Encoder:
The input is a sequence of phonemes IDs.
- Input: [Batch, Phoneme_Length] (eg: "Cat" -> [k, ae, t], length=3)
- Operation:
    1. Embedding: Convert IDs to vectors
    2. FFT Blocks: It consists of self-attention and 1D convolutions
        - Self-attention captures global context
        - 1D convolutions capture local context
- Output: Hidden states $H_{phoneme}$ [Batch, Phoneme_Length, Hidden_Size]

#### The Length Regulator:
This is the component that enables parallel generation. It solves the Length Mismatch Problem
- Phoneme sequence length: Small
- Mel-Spectrogram length: Large

We cannot feed a length-50 sequence into a network and expect a length-500 output without a loop unless we explicitly resize the intermediate representation.

How it works:
- Duration Predictor: A small sub-network predicts how many Mel-frames each phoneme corresponds to.
    - Input: $H_{phoneme}$ (from Encoder).
    - Output: Integers [d_1, d_2, d_3, ...] (e.g., [10, 24, 15]).
- Expansion: The Length Regulator takes $H_{phoneme}$ and repeats each vector $d_i$ times.

#### The Decoder:
Now we have $H_{mel}$, which is roughly aligned with time but just contains repeated phoneme information. The Decoder turns this rough sketch into detailed acoustic features.
- Input: $H_{mel}$ (Expanded sequence).
- Structure: Stack of FFT Blocks (same as Encoder).
    - Because the sequence is already expanded to the correct length, the Decoder just applies self-attention across the whole timeline.
    - It refines the repeated vectors. Frame 3 and Frame 4 might both come from h_ae, but the Position Embeddings and Self-Attention allow the model to make Frame 3 the "start of the vowel" and Frame 4 the "middle of the vowel."
- Output: The final Mel-Spectrogram [Batch, Mel_Length, Mel_Channels].

## Phenome Duration Predictor:

This answers the question: "How long should I hold this sound?" without needing to see what was generated 5 milliseconds ago.

Fastspeech relies on a pre-trained "Teacher" model (usually Tactron 2) to figure out the timing.

1. Knowledge distilation

Step A: Train the teacher

Train a standard autoregressive Tactron2 model on the dataset. It learns an Attention Matrix (Alignment) during training. This matrix visualizes which text characters the model is looking at while generating each audio frame.

Step B: Extract Duration (The "Hard" Alignment)

We look at the Teacher's attention map for every training sentence.

The attention map is usually "soft" (fuzzy). We convert it into a "hard" alignment using a technique called Monotonic Alignment Search (MAS) or simple argmax heuristic.

Basically, we count how many Mel-frames "attend" to each phoneme.

- Phoneme: [H, e, l, l, o]
- Teacher's Attention:
    - Frames 1-10 looked mostly at "H".
    - Frames 11-22 looked mostly at "e".
- Derived Label: Duration = [10, 12, ...]

These extracted integers become the Ground Truth Labels for training the Duration Predictor.

2. The Architecture
The Duration Predictor is a small regression network stacked on top of the Encoder.
- Input: Encoder Hidden States $H_{phoneme}$ (Shape: [Batch, Phoneme_Length, 384])
- Layers:
    1. Conv1D: Kernel size 3, 256 filters, ReLU activation. (Captures context like "vowels are longer before voiced consonants").
    2. LayerNorm + Dropout: For stabilization.
    3. Conv1D: Another layer to refine features.
    4. LayerNorm + Dropout.
    5. Linear Projection: Projects 256 dimensions down to 1 scalar (the duration).
- Output: A scalar value for each phoneme (Shape: [Batch, Phoneme_Length, 1]).

3. Training vs Inference Strat

This is a critical implementation detail.

During Training (Teacher Forcing)

When training FastSpeech, we want the Decoder to learn good audio generation. If the Duration Predictor guesses wrong, the length of the Mel-spectrogram will be wrong, and the loss calculation will fail (dimension mismatch).
- Strategy: We use the Ground Truth Durations (from the Teacher) to perform the expansion in the Length Regulator.
- Side Task: Simultaneously, we train the Duration Predictor to minimize the error between its prediction and the Ground Truth.
- Loss: Mean Squared Error (MSE) between Predicted_Duration and Ground_Truth_Duration (often in log-scale to guarantee positive values).

During Inference (The "Free" Run)

We don't have ground truth anymore.
- Strategy: We feed the text to the Encoder, run the Duration Predictor to get - estimated durations (rounded to nearest integer), and use those to drive the Length Regulator.

4. Conclusion (kind-off)

In autogressive models, attention is calculated on the fly. If the attention mechanism gets "confused" (distracted by noise or complex words), it might skip ahead or loop back causing skipping/stuttering.

In FastSpeech, the duration is predicted once as a hard integer
- If the model predicts "H" duration is 5, it will generate exactly 5 frames for "H".
- It physically cannot skip the phoneme (unless duration is predicted as 0) or get stuck in a loop.

It also gives us free speed control. Meaning since the length regulator takes a list of integers, we can multiple them by a factor of $\alpha$ to before expansion.

## Length Regulator:

The Length Regulator is the structural pivot of the entire architecture. It is "bridge" that allows us to perform the dimension transformation from the Phoneme Domain ($N$ length) to the Mel-Spectrogram Domain ($T$ length).

Without this component, we cannot conect the encoder (text processing) to the Decoder (audio generation) in a non-autoregressive way.

1. The Core Mechanism: Hard Upsampling

The operation is mathematically simple but structurally profound. It performs repitition based on duration. 

Let
- $H_{pho} = [h1_, h_2, h_3,....., h_N]$ be the sequence of phoneme hidden states.
- $\mathcal{D} = [d_1, d_2, d_3,......,d_N]$ be the sequence of durations (integers), where $\sum \mathcal{D} = T_{total}$.

The length regulator expands $H_{pho}$ to $H_{mel}$ by repeating vector $h_i$ $d_i$ times.

$$ LR(H_{pho}, \mathcal{D}) = [\underbrace{h_1, ..., h_1}{d_1}, \underbrace{h_2, ..., h_2}{d_2}, ..., \underbrace{h_N, ..., h_N}{d_N}] $$

2. Visualizing the Tensor Operations

Imagine a batch size of 1.

Input (Encoder Output):
- Shape: (1, 3, 384) -> 3 phonemes (e.g., "C", "a", "t"), 384-dim vector.
- Vectors: [v_C, v_a, v_t]

Duration Input:
- Values: [2, 3, 1] -> "C" takes 2 ticks, "a" takes 3, "t" takes 1.

Operation:
1. Take v_C, replicate 2x -> [v_C, v_C]
2. Take v_a, replicate 3x -> [v_a, v_a, v_a]
3. Take v_t, replicate 1x -> [v_t]
4. Concatenate

Output (Decoder Input):
- Shape: (1, 6, 384)
- Sequence: [v_C, v_C, v_a, v_a, v_a, v_t]

Now, the Decoder has a sequence of length 6. It doesn't know that position 1 and 2 are identical copies initially. It just sees a sequence. The Positional Encodings added after this step are crucial—they tell the Decoder, "This is the first millisecond of 'C', and this is the second millisecond of 'C'."

3. Controlling Voice Speed (via $\alpha$)
This is where the "Controllable" part of the paper title comes in. Because the expansion is explicit, we can manipulate it mathematically.

We introduce a hyperparameter $\alpha$ (Speed Ratio).

$$ \mathcal{D}{new} = \text{Round}(\mathcal{D}{original} \times \frac{1}{\alpha}) $$

- $\alpha = 1.0$: Normal speed.
- $\alpha = 1.3$: Fast speech. We shorten durations. Round([2, 3, 1] * 0.76).
- $\alpha = 0.5$: Slow motion. We lengthen durations.

Note: We divide by $\alpha$ usually because in signal processing, higher speed means shorter time.
- To double speed (2x), duration must be halved (0.5x).

4. Why "Hard" vs "Soft" Alignment?

- Tacotron (Soft): The model decides at every step "should I move to the next phoneme?" with a probability distribution. It's fuzzy.
- FastSpeech (Hard): The Length Regulator makes a binary decision. Frame 5 is Phoneme A. Frame 6 is Phoneme B. No ambiguity.

This rigidity is the reason FastSpeech is so robust. It physically cannot babble because the output length is strictly bounded by $\sum \mathcal{D}$. It cannot generate infinite audio frames because it pre-calculates the exact finish line before starting generation.
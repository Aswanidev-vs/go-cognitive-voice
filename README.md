<p align="center">
  <img src="logo.svg" width="200" alt="GCV Logo"/>
</p>

<h1 align="center">go-cognitive-voice</h1>

<p align="center">
  <a href="docs/index.html">Documentation</a>
</p>

<p align="center">
  <strong>Pure Go Speaker Detection & Identification</strong><br>
  <em>Zero CGo. Zero ML frameworks. Just math.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Go-1.22+-00ADD8?style=for-the-badge&logo=go&logoColor=white" alt="Go"/>
  <img src="https://img.shields.io/badge/CGo-None-brightgreen?style=for-the-badge" alt="No CGo"/>
  <img src="https://img.shields.io/badge/Dependencies-2-blueviolet?style=for-the-badge" alt="Minimal deps"/>
</p>

---

## What is this?

A speaker detection and identification library written in Go. Feed it audio files, it tells you who's talking.

```bash
go get github.com/Aswanidev-vs/go-cognitive-voice
```

## 30-Second Integration

```go
package main

import (
    "fmt"
    "github.com/Aswanidev-vs/go-cognitive-voice/pkg/gcv"
)

func main() {
    engine := gcv.NewEngine(gcv.DefaultConfig())

    // Enroll speakers (any number of WAV/MP3 files per speaker)
    engine.EnrollSpeaker("alice", "Alice", []string{"alice.wav"})
    engine.EnrollSpeaker("bob",   "Bob",   []string{"bob.wav"})

    // Identify who's talking
    result, _ := engine.Identify("mystery.wav")
    if result.Identified {
        fmt.Printf("%s is talking (confidence: %.0f%%)\n", result.SpeakerName, result.Score*100)
    }
}
```

That's it. 10 lines of code.

## How It Works

```
Audio (.wav/.mp3)
    │
    ▼
┌─────────────────┐
│  Audio Decoder   │  WAV/MP3 → mono float64 samples
└────────┬────────┘
         ▼
┌─────────────────┐
│  Pre-emphasis    │  High-pass filter
│  + Framing       │  25ms frames, 10ms hop
│  + Hamming       │  Windowing
└────────┬────────┘
         ▼
┌─────────────────┐
│  FFT → Mel       │  Frequency analysis
│  Filter Bank     │  26 mel-spaced filters
└────────┬────────┘
         ▼
┌─────────────────┐
│  DCT → MFCC      │  13 coeffs + energy + deltas
│  (42 features)   │  = voice fingerprint
└────────┬────────┘
         ▼
┌─────────────────┐
│  Cosine +        │  Compare against
│  Mahalanobis     │  enrolled speakers
└────────┬────────┘
         ▼
    Identity
```

## API Reference

### Core Functions

| Function | What it does |
|----------|-------------|
| `NewEngine(cfg)` | Create an engine |
| `EnrollSpeaker(id, name, files)` | Register a speaker from audio files |
| `Identify(audioPath)` | Find who's talking |
| `Detect(audioPath, speakerID)` | Verify if it's a specific person |
| `RemoveSpeaker(id)` | Delete a speaker |
| `ListSpeakers()` | List all enrolled speakers |
| `SaveDatabase(dir)` | Persist to disk |
| `LoadDatabase(dir)` | Load from disk |

### Configuration

```go
cfg := gcv.DefaultConfig()

// Tuning the match threshold:
//   0.5  = lenient (default, good for clean audio)
//   0.7  = moderate (recommended for real-world)
//   0.85 = strict (fewer false positives)
cfg.MatchThreshold = 0.85

// Training method:
//   "centroid" — fast, default
//   "gmm"      — better for noisy audio
//   "dtw"      — best for variable-length
cfg.TrainMethod = "centroid"

engine := gcv.NewEngine(cfg)
```

### Working with Audio

```go
// From files (easiest)
result, _ := engine.Identify("audio.wav")

// From raw samples (for microphone/streaming input)
samples, sampleRate, _ := gcv.ReadAudio("audio.wav")
result, _ := engine.IdentifyFromSamples(samples, sampleRate)

// From pre-extracted features (for custom pipelines)
features, _ := engine.ExtractFeatures("audio.wav")
result := engine.IdentifyFromFeatures(features)
```

### Feature Extraction Only

```go
features := gcv.ExtractMFCC(samples, sampleRate)
// features: [][]float64 — [numFrames][42]
//   [0:14]  = MFCC (13 coefficients + log energy)
//   [14:28] = delta (1st derivative)
//   [28:42] = delta-delta (2nd derivative)
```

### Streaming Audio

For live/microphone/websocket input, use `StreamBuffer`. It accumulates audio in chunks and identifies as audio arrives.

```go
engine := gcv.NewEngine(gcv.DefaultConfig())
engine.EnrollSpeaker("alice", "Alice", []string{"alice.wav"})

// Identify every 1 second of audio
buf := engine.NewStreamBuffer(16000, 1.0)

// Feed chunks from your audio source
for chunk := range audioChan {
    buf.Write(chunk)

    if result, ok := buf.Identify(); ok {
        fmt.Printf("%s (%.1f%%)\n", result.SpeakerName, result.Score*100)
    }
}

// Final identification on all accumulated audio
result := buf.IdentifyFull()
```

| Method | When to use |
|--------|-------------|
| `Write(samples)` | Push audio chunks from mic/websocket/file |
| `Identify()` | Runs when enough audio accumulated (returns `ok=false` otherwise) |
| `IdentifyFull()` | Use at stream end for final ID on all audio |
| `Reset()` | Clear buffer between speakers |
| `Duration()` | Check how much audio is buffered |

## Audio Requirements

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Format | WAV (16-bit PCM) or MP3 | Any sample rate works |
| Sample rate | 16000 Hz | Standard for speech |
| Duration | 1-10 seconds | Longer = more accurate |
| Channels | Mono or stereo | Stereo is auto-mixed to mono |
| Quantity | 2-5 files per speaker | More files = better accuracy |

## Scoring Guide

Scores range from 0.0 to 1.0:

| Score | Meaning |
|-------|---------|
| 1.00 | Perfect self-match |
| 0.85+ | Strong match (same speaker) |
| 0.70-0.85 | Weak match (possibly same speaker, different conditions) |
| 0.50-0.70 | Low confidence (different speaker, similar voice) |
| < 0.50 | No match (different speaker) |

**Threshold recommendations:**
- `0.5` — default, accepts most matches
- `0.7` — good for production use
- `0.85` — strict, rejects similar-sounding speakers

## Examples

See the [`examples/`](examples/) directory:

```bash
# Basic usage
go run examples/basic/main.go

# Advanced: persistence, threshold tuning, verification
go run examples/advanced/main.go

# MFCC feature extraction
go run examples/mfcc/main.go audio.wav

# Streaming audio (real-time identification)
go run examples/streaming/main.go audio.wav
```

## CLI

```bash
# Build
go build -o gcv ./cmd/gcv/

# Enroll (re-enrolling same ID replaces the old model)
./gcv enroll alice "Alice" alice_01.wav alice_02.wav

# Identify
./gcv identify mystery.wav

# Verify
./gcv detect mystery.wav alice

# List
./gcv list

# Show features
./gcv features audio.wav
```

## Architecture

```
go-cognitive-voice/
├── cmd/gcv/              CLI tool
├── examples/             Runnable examples
├── internal/
│   ├── audio/            WAV + MP3 decoding, framing, windowing
│   ├── dsp/              FFT, mel filters, MFCC extraction
│   ├── matcher/          DTW, cosine/Euclidean/Mahalanobis distance
│   └── speaker/          Speaker models, GMM, engine, persistence
├── pkg/gcv/              Public API (import this)
├── logo.svg
└── README.md
```

## Dependencies

Only 1 external package:

| Package | Purpose | CGo? |
|---------|---------|------|
| [`gopxl/beep`](https://github.com/gopxl/beep) | Audio decoding (WAV + MP3) | No |

Everything else — FFT, mel filter banks, MFCC, DTW, GMM, neural networks, i-vectors, PLDA, RASTA, PLP, pitch detection, VAD, matrix math — is implemented from scratch.

## Advanced Features

Enable neural network embeddings and i-vector + PLDA for better accuracy. **No manual training required** — the network trains automatically when you enroll speakers:

```go
cfg := gcv.DefaultConfig()
cfg.UseNeural = true      // custom neural network (trained from scratch)
cfg.UseIVector = true     // i-vector + PLDA scoring
cfg.UseRASTA = true       // noise-robust features
cfg.UseVAD = true         // voice activity detection
cfg.MatchThreshold = 0.7

engine := gcv.NewEngine(cfg)

// Training happens automatically inside EnrollSpeaker.
// Provide audio files, the network learns from them.
engine.EnrollSpeaker("alice", "Alice", []string{"alice_01.wav", "alice_02.wav"})
engine.EnrollSpeaker("bob", "Bob", []string{"bob_01.wav"})

// Now identify — uses the trained neural network
result, _ := engine.Identify("mystery.wav")
```

**How it works under the hood:**
1. `EnrollSpeaker` extracts MFCC features from your audio files
2. When 2+ speakers are enrolled, the engine **automatically trains** a neural network on all speakers' features using contrastive learning
3. It also trains an i-vector + PLDA model if `UseIVector` is enabled
4. Both models are used together for scoring (60%% neural, 40%% i-vector)

You never call `Train()` — it's all automatic.

### What's built from scratch (no pre-trained models)

| Component | What it does | When it runs |
|-----------|-------------|-------------|
| **Neural Network** | Custom feedforward net with contrastive loss | Auto-trains on 2nd+ enrollment |
| **i-vector** | GMM-UBM + total variability matrix | Auto-trains on 2nd+ enrollment |
| **PLDA** | Probabilistic LDA scoring | Auto-trains on 2nd+ enrollment |
| **RASTA** | Bandpass filtering for noise robustness | Applied during feature extraction |
| **PLP** | Perceptual Linear Prediction (alternative to MFCC) | Set `UsePLP: true` |
| **Pitch Detection** | Autocorrelation-based F0 tracking |
| **VAD** | Voice Activity Detection | Applied during feature extraction |
| **Spectral Subtraction** | Noise reduction | Available via `dsp.SpectralSubtraction` |

## Comparison with Python Alternatives

| | GCV (this) | speechbrain | resemblyzer | pyannote-audio |
|---|---|---|---|---|
| **Language** | Go | Python | Python | Python |
| **Algorithm** | MFCC/GMM + neural + i-vector | ECAPA-TDNN (neural) | d-vector (neural) | PyanNet (neural) |
| **Dependencies** | 1 (beep) | PyTorch + 10+ pkgs | PyTorch + 5+ pkgs | PyTorch + 8+ pkgs |
| **Install size** | ~2 MB | ~2 GB+ | ~2 GB+ | ~2 GB+ |
| **CGo required** | No | N/A | N/A | N/A |
| **Streaming** | Yes | Yes | No | Yes |
| **License** | MIT | Apache 2.0 | MIT | MIT |
| **Diarization** | No | Yes | No | Yes |
| **Pre-trained models** | None (trains from scratch) | Required | Required | Required |
| **Training data needed** | Your enrollment audio | Pre-trained on VoxCeleb | Pre-trained on VoxCeleb | Pre-trained on VoxCeleb |

**When to use GCV:**
- You need speaker ID in a Go service (no Python runtime)
- You want minimal dependencies and fast startup
- You need a lightweight embedded solution
- You want to train on your own data without pre-trained models

**When to use Python alternatives:**
- You need speaker diarization (who spoke when)
- You're building an ML research pipeline
- Python runtime is acceptable

**Key difference:** Python libraries use pre-trained neural networks (trained on 1M+ utterances from VoxCeleb). GCV trains from scratch on your enrollment data — no downloads, no model files, no PyTorch.

## Testing

```bash
go test ./... -v
go test -race ./...    # with race detector
go vet ./...           # static analysis
```

## License

MIT

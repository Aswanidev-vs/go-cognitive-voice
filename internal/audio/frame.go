package audio

import "math"

// Framing parameters
const (
	DefaultFrameLenMs  = 25.0  // 25ms frames
	DefaultFrameShiftMs = 10.0 // 10ms shift (hop)
	DefaultPreEmph     = 0.97  // pre-emphasis coefficient
)

// FrameConfig holds framing parameters.
type FrameConfig struct {
	FrameLenMs   float64
	FrameShiftMs float64
	PreEmphCoeff float64
}

// DefaultFrameConfig returns sensible defaults for speech processing.
func DefaultFrameConfig() FrameConfig {
	return FrameConfig{
		FrameLenMs:   DefaultFrameLenMs,
		FrameShiftMs: DefaultFrameShiftMs,
		PreEmphCoeff: DefaultPreEmph,
	}
}

// PreEmphasis applies a high-pass filter to emphasize higher frequencies.
func PreEmphasis(samples []float64, coeff float64) []float64 {
	if len(samples) == 0 {
		return nil
	}
	out := make([]float64, len(samples))
	out[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		out[i] = samples[i] - coeff*samples[i-1]
	}
	return out
}

// Frame splits audio into overlapping frames.
// Returns a 2D slice: [numFrames][frameLen].
func Frame(samples []float64, sampleRate int, cfg FrameConfig) [][]float64 {
	if len(samples) == 0 {
		return nil
	}

	frameLen := int(cfg.FrameLenMs * float64(sampleRate) / 1000.0)
	frameShift := int(cfg.FrameShiftMs * float64(sampleRate) / 1000.0)

	if frameShift < 1 {
		frameShift = 1
	}
	if frameLen < 2 {
		frameLen = 2
	}

	// If signal is shorter than one frame, pad to frame length
	if len(samples) < frameLen {
		padded := make([]float64, frameLen)
		copy(padded, samples)
		return [][]float64{padded}
	}

	// Pad signal if needed
	padLen := frameLen - len(samples)%frameShift
	if padLen == frameShift {
		padLen = 0
	}
	padded := make([]float64, len(samples)+padLen)
	copy(padded, samples)

	numFrames := (len(padded)-frameLen)/frameShift + 1
	frames := make([][]float64, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * frameShift
		frame := make([]float64, frameLen)
		copy(frame, padded[start:start+frameLen])
		frames[i] = frame
	}

	return frames
}

// HammingWindow applies a Hamming window to a frame in-place.
func HammingWindow(frame []float64) {
	n := len(frame)
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		frame[i] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(i)/float64(n-1))
	}
}

// HanningWindow applies a Hanning window to a frame in-place.
func HanningWindow(frame []float64) {
	n := len(frame)
	if n <= 1 {
		return
	}
	for i := 0; i < n; i++ {
		frame[i] *= 0.5 * (1.0 - math.Cos(2*math.Pi*float64(i)/float64(n-1)))
	}
}

// ProcessFrames applies pre-emphasis, framing, and windowing in one call.
func ProcessFrames(samples []float64, sampleRate int, cfg FrameConfig) [][]float64 {
	// Pre-emphasis
	preemphed := PreEmphasis(samples, cfg.PreEmphCoeff)

	// Frame
	frames := Frame(preemphed, sampleRate, cfg)

	// Window each frame
	for _, frame := range frames {
		HammingWindow(frame)
	}

	return frames
}

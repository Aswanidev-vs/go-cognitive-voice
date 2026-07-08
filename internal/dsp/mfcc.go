package dsp

import "math"

// MFCCConfig holds parameters for MFCC extraction.
type MFCCConfig struct {
	NumMFCC     int     // Number of MFCC coefficients (typically 13)
	NumFilters  int     // Number of mel filters (typically 26)
	FFTSize     int     // FFT size (power of 2)
	SampleRate  int     // Audio sample rate
	LowFreq     float64 // Lowest frequency for mel filter bank
	HighFreq    float64 // Highest frequency (0 = Nyquist)
	IncludeEnergy bool  // Whether to include log energy as 0th coefficient
}

// DefaultMFCCConfig returns sensible defaults for speech processing.
func DefaultMFCCConfig(sampleRate int) MFCCConfig {
	fftSize := 512
	if sampleRate >= 16000 {
		fftSize = 512
	}
	return MFCCConfig{
		NumMFCC:      13,
		NumFilters:   26,
		FFTSize:      fftSize,
		SampleRate:   sampleRate,
		LowFreq:      0,
		HighFreq:     float64(sampleRate) / 2.0,
		IncludeEnergy: true,
	}
}

// MFCCResult holds the complete MFCC output for an audio signal.
type MFCCResult struct {
	Coefficients [][]float64 // [numFrames][numCoeffs]
	Deltas       [][]float64 // Delta features [numFrames][numCoeffs]
	DeltaDeltas  [][]float64 // Delta-delta features [numFrames][numCoeffs]
}

// ExtractMFCC extracts MFCC features from framed, windowed audio.
// Input frames should already be pre-emphasized, framed, and windowed.
func ExtractMFCC(frames [][]float64, cfg MFCCConfig) MFCCResult {
	numFrames := len(frames)

	// Create mel filter bank
	filters := MelFilterBank(cfg.NumFilters, cfg.FFTSize, cfg.SampleRate, cfg.LowFreq, cfg.HighFreq)

	// DCT matrix produces NumMFCC coefficients; energy is prepended separately
	dctMatrix := makeDCTMatrix(cfg.NumFilters, cfg.NumMFCC)

	result := make([][]float64, numFrames)

	for i, frame := range frames {
		// Compute power spectrum
		psd := PowerSpectrum(frame)

		// Apply mel filter bank
		melEnergies := ApplyMelFilterBank(psd, filters)

		// Apply DCT to get MFCC
		mfcc := applyDCT(melEnergies, dctMatrix)

		// Prepend log energy as 0th coefficient if requested
		if cfg.IncludeEnergy {
			energy := LogEnergy(psd)
			mfccWithEnergy := make([]float64, len(mfcc)+1)
			mfccWithEnergy[0] = energy
			copy(mfccWithEnergy[1:], mfcc)
			result[i] = mfccWithEnergy
		} else {
			result[i] = mfcc
		}
	}

	// Compute deltas
	deltas := computeDeltas(result)
	deltaDeltas := computeDeltas(deltas)

	return MFCCResult{
		Coefficients: result,
		Deltas:       deltas,
		DeltaDeltas:  deltaDeltas,
	}
}

// ExtractMFCCFromRaw extracts MFCC directly from raw audio samples.
func ExtractMFCCFromRaw(samples []float64, sampleRate int) MFCCResult {
	cfg := DefaultMFCCConfig(sampleRate)

	// Frame and window
	frameCfg := FrameConfig{
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.97,
	}
	frames := ProcessFrames(samples, sampleRate, frameCfg)

	return ExtractMFCC(frames, cfg)
}

// FrameConfig for framing (mirrors audio.FrameConfig but internal).
type FrameConfig struct {
	FrameLenMs   float64
	FrameShiftMs float64
	PreEmphCoeff float64
}

// ProcessFrames applies pre-emphasis, framing, and windowing.
func ProcessFrames(samples []float64, sampleRate int, cfg FrameConfig) [][]float64 {
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

	// Pre-emphasis
	preemphed := make([]float64, len(samples))
	preemphed[0] = samples[0]
	for i := 1; i < len(samples); i++ {
		preemphed[i] = samples[i] - cfg.PreEmphCoeff*samples[i-1]
	}

	// If signal is shorter than one frame, pad to frame length
	if len(preemphed) < frameLen {
		padded := make([]float64, frameLen)
		copy(padded, preemphed)
		frame := make([]float64, frameLen)
		copy(frame, padded)
		n := len(frame)
		for j := 0; j < n; j++ {
			frame[j] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(j)/float64(n-1))
		}
		return [][]float64{frame}
	}

	// Pad signal
	padLen := frameLen - len(preemphed)%frameShift
	if padLen == frameShift {
		padLen = 0
	}
	padded := make([]float64, len(preemphed)+padLen)
	copy(padded, preemphed)

	numFrames := (len(padded)-frameLen)/frameShift + 1
	frames := make([][]float64, numFrames)

	for i := 0; i < numFrames; i++ {
		start := i * frameShift
		frame := make([]float64, frameLen)
		copy(frame, padded[start:start+frameLen])

		// Hamming window
		n := len(frame)
		for j := 0; j < n; j++ {
			frame[j] *= 0.54 - 0.46*math.Cos(2*math.Pi*float64(j)/float64(n-1))
		}

		frames[i] = frame
	}

	return frames
}

// makeDCTMatrix creates a Type-II DCT matrix.
func makeDCTMatrix(inputSize, outputSize int) [][]float64 {
	matrix := make([][]float64, outputSize)
	for i := 0; i < outputSize; i++ {
		matrix[i] = make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			matrix[i][j] = math.Cos(math.Pi * float64(i) * (2*float64(j) + 1) / (2 * float64(inputSize)))
		}
	}
	return matrix
}

// applyDCT applies a DCT matrix to a vector.
func applyDCT(x []float64, matrix [][]float64) []float64 {
	n := len(matrix)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < len(x) && j < len(matrix[i]); j++ {
			sum += x[j] * matrix[i][j]
		}
		result[i] = sum
	}
	return result
}

// computeDeltas computes delta (differential) features.
// Uses a simple two-frame difference: delta[t] = (c[t+1] - c[t-1]) / 2
func computeDeltas(features [][]float64) [][]float64 {
	if len(features) < 2 {
		deltas := make([][]float64, len(features))
		for i := range deltas {
			deltas[i] = make([]float64, len(features[i]))
		}
		return deltas
	}

	numFrames := len(features)
	numCoeffs := len(features[0])
	deltas := make([][]float64, numFrames)

	for t := 0; t < numFrames; t++ {
		deltas[t] = make([]float64, numCoeffs)
		for c := 0; c < numCoeffs; c++ {
			if t == 0 {
				deltas[t][c] = features[1][c] - features[0][c]
			} else if t == numFrames-1 {
				deltas[t][c] = features[numFrames-1][c] - features[numFrames-2][c]
			} else {
				deltas[t][c] = (features[t+1][c] - features[t-1][c]) / 2.0
			}
		}
	}

	return deltas
}

// ConcatenateFeatures concatenates MFCC, deltas, and delta-deltas into a single feature matrix.
func ConcatenateFeatures(result MFCCResult) [][]float64 {
	numFrames := len(result.Coefficients)
	if numFrames == 0 {
		return nil
	}

	numCoeffs := len(result.Coefficients[0])
	numDeltas := 0
	if result.Deltas != nil && len(result.Deltas) > 0 {
		numDeltas = len(result.Deltas[0])
	}
	numDeltaDeltas := 0
	if result.DeltaDeltas != nil && len(result.DeltaDeltas) > 0 {
		numDeltaDeltas = len(result.DeltaDeltas[0])
	}

	totalCoeffs := numCoeffs + numDeltas + numDeltaDeltas
	features := make([][]float64, numFrames)

	for t := 0; t < numFrames; t++ {
		features[t] = make([]float64, totalCoeffs)
		idx := 0

		// Copy MFCC
		copy(features[t][idx:], result.Coefficients[t])
		idx += numCoeffs

		// Copy deltas
		if result.Deltas != nil && t < len(result.Deltas) {
			copy(features[t][idx:], result.Deltas[t])
			idx += numDeltas
		}

		// Copy delta-deltas
		if result.DeltaDeltas != nil && t < len(result.DeltaDeltas) {
			copy(features[t][idx:], result.DeltaDeltas[t])
			idx += numDeltaDeltas
		}
	}

	return features
}

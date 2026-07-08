package dsp

import "math"

// HzToMel converts frequency in Hz to Mel scale.
func HzToMel(hz float64) float64 {
	return 2595.0 * math.Log10(1.0+hz/700.0)
}

// MelToHz converts Mel scale to frequency in Hz.
func MelToHz(mel float64) float64 {
	return 700.0 * (math.Pow(10.0, mel/2595.0) - 1.0)
}

// MelFilterBank computes triangular mel-spaced filter bank weights.
// Returns a matrix of shape [numFilters][fftSize/2+1].
//
// Parameters:
//   - numFilters: number of mel filters (typically 26 for MFCC)
//   - fftSize: FFT size (power of 2)
//   - sampleRate: audio sample rate in Hz
//   - lowFreq: lowest frequency (typically 0)
//   - highFreq: highest frequency (typically sampleRate/2)
func MelFilterBank(numFilters, fftSize, sampleRate int, lowFreq, highFreq float64) [][]float64 {
	if lowFreq <= 0 {
		lowFreq = 0
	}
	if highFreq <= 0 {
		highFreq = float64(sampleRate) / 2.0
	}

	fftBins := fftSize/2 + 1

	// Convert to mel scale
	melLow := HzToMel(lowFreq)
	melHigh := HzToMel(highFreq)

	// Create evenly spaced points in mel scale
	melPoints := make([]float64, numFilters+2)
	for i := range melPoints {
		melPoints[i] = melLow + float64(i)*(melHigh-melLow)/float64(numFilters+1)
	}

	// Convert back to Hz
	hzPoints := make([]float64, len(melPoints))
	for i, m := range melPoints {
		hzPoints[i] = MelToHz(m)
	}

	// Convert Hz to FFT bin indices
	binPoints := make([]float64, len(hzPoints))
	for i, hz := range hzPoints {
		binPoints[i] = float64(fftBins-1) * hz / (float64(sampleRate) / 2.0)
	}

	// Create filter bank
	filters := make([][]float64, numFilters)
	for i := 0; i < numFilters; i++ {
		filters[i] = make([]float64, fftBins)
		left := binPoints[i]
		center := binPoints[i+1]
		right := binPoints[i+2]

		for j := 0; j < fftBins; j++ {
			fj := float64(j)
			if fj >= left && fj <= center {
				if center != left {
					filters[i][j] = (fj - left) / (center - left)
				}
			} else if fj > center && fj <= right {
				if right != center {
					filters[i][j] = (right - fj) / (right - center)
				}
			}
		}
	}

	return filters
}

// ApplyMelFilterBank applies mel filter bank to a power spectrum.
// Returns the log energy of each filter band.
func ApplyMelFilterBank(powerSpectrum []float64, filters [][]float64) []float64 {
	numFilters := len(filters)
	energies := make([]float64, numFilters)

	for i, filter := range filters {
		sum := 0.0
		// Ensure we don't exceed the power spectrum length
		limit := len(powerSpectrum)
		if len(filter) < limit {
			limit = len(filter)
		}
		for j := 0; j < limit; j++ {
			sum += powerSpectrum[j] * filter[j]
		}
		// Log with floor to avoid log(0)
		if sum < 1e-20 {
			energies[i] = -20.0
		} else {
			energies[i] = math.Log(sum)
		}
	}

	return energies
}

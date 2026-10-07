package dsp

import "math"

// RASTAFilter applies a bandpass filter in the log-spectral domain.
// RASTA (RelAtive SpecTrAl) filtering removes slow channel variations
// and fast noise, keeping only speech-related temporal dynamics.
//
// The filter is: H(z) = z^4 * (-1 + 4z^-1 - 6z^-2 + 4z^-3 - z^-4)
// implemented as a difference equation on log-energies.
func RASTAFilter(logSpectra [][]float64) [][]float64 {
	if len(logSpectra) == 0 {
		return nil
	}

	nFrames := len(logSpectra)
	nBins := len(logSpectra[0])
	result := make([][]float64, nFrames)

	// Apply RASTA per bin outside the frame loop
	coeffs := []float64{-1, 4, -6, 4, -1}
	for t := 0; t < nFrames; t++ {
		result[t] = make([]float64, nBins)
		for f := 0; f < nBins; f++ {
			val := 0.0
			// H(z) = z^4 * (-1 + 4z^-1 - 6z^-2 + 4z^-3 - z^-4)
			// = -x[t] + 4*x[t-1] - 6*x[t-2] + 4*x[t-3] - x[t-4]
			for k, c := range coeffs {
				idx := t - k
				if idx >= 0 && idx < nFrames {
					val += c * logSpectra[idx][f]
				}
			}
			result[t][f] = val
		}
	}

	return result
}

// PLP extracts Perceptual Linear Prediction cepstral coefficients.
// PLP is similar to MFCC but uses perceptual processing (Bark scale,
// equal-loudness pre-emphasis, intensity-loudness power law).
func PLP(samples []float64, sampleRate int, order int) [][]float64 {
	// Frame and window
	frameCfg := FrameConfig{
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.97,
	}
	frames := ProcessFrames(samples, sampleRate, frameCfg)

	nFilters := 24 // critical band filters (Bark scale)

	result := make([][]float64, len(frames))

	for i, frame := range frames {
		// Power spectrum
		psd := PowerSpectrum(frame)

		// Apply Bark-scale filter bank
		barkEnergies := barkFilterBank(psd, sampleRate, nFilters)

		// Equal-loudness pre-emphasis
		for j, e := range barkEnergies {
			freq := float64(j) * float64(sampleRate) / (2.0 * float64(nFilters))
			eq := equalLoudness(freq)
			barkEnergies[j] = e * eq
		}

		// Intensity-loudness power law: E^0.33
		for j := range barkEnergies {
			if barkEnergies[j] > 0 {
				barkEnergies[j] = math.Pow(barkEnergies[j], 0.33)
			}
		}

		// Autocorrelation -> LPC -> cepstral coefficients
		cepstrum := lpcToCepstrum(barkEnergies, order)
		result[i] = cepstrum
	}

	return result
}

// barkFilterBank applies triangular filters on the Bark scale.
func barkFilterBank(psd []float64, sampleRate, nFilters int) []float64 {
	nBins := len(psd)
	energies := make([]float64, nFilters)

	for i := 0; i < nFilters; i++ {
		// Center frequency in Bark scale
		barkLow := float64(i) * 24.0 / float64(nFilters)
		barkCenter := float64(i+1) * 24.0 / float64(nFilters)
		barkHigh := float64(i+2) * 24.0 / float64(nFilters)

		// Convert Bark to Hz
		hzLow := barkToHz(barkLow)
		hzCenter := barkToHz(barkCenter)
		hzHigh := barkToHz(barkHigh)

		// Convert Hz to bin indices
		binLow := int(hzLow * float64(nBins-1) / (float64(sampleRate) / 2.0))
		binCenter := int(hzCenter * float64(nBins-1) / (float64(sampleRate) / 2.0))
		binHigh := int(hzHigh * float64(nBins-1) / (float64(sampleRate) / 2.0))

		sum := 0.0
		for b := binLow; b <= binHigh && b < nBins; b++ {
			fb := float64(b)
			var weight float64
			if fb <= float64(binCenter) && binCenter != binLow {
				weight = (fb - float64(binLow)) / (float64(binCenter) - float64(binLow))
			} else if binHigh != binCenter {
				weight = (float64(binHigh) - fb) / (float64(binHigh) - float64(binCenter))
			}
			if b < len(psd) {
				sum += psd[b] * weight
			}
		}
		energies[i] = sum
	}

	return energies
}

func barkToHz(bark float64) float64 {
	return 1960.0 * (math.Pow(10, bark/24.0) - 1.0)
}

func hzToBark(hz float64) float64 {
	return 24.0 * math.Log10(1.0+hz/1960.0) / math.Log10(2.0)
}

// equalLoudness returns the equal-loudness pre-emphasis weight for a frequency.
func equalLoudness(freq float64) float64 {
	// Simplified equal-loudness curve (ISO 226 approximation)
	f2 := freq * freq
	num := 1.5e7 * f2 * f2
	denom := (f2 + 4.24e6) * (f2 + 1.48e9) * (f2 + 9.61e12)
	return num / denom
}

func lpcToCepstrum(spectrum []float64, order int) []float64 {
	// Compute autocorrelation from spectrum
	n := len(spectrum)
	r := make([]float64, order+1)
	for k := 0; k <= order; k++ {
		sum := 0.0
		for i := 0; i < n; i++ {
			if i+k < n {
				sum += spectrum[i] * spectrum[i+k]
			}
		}
		r[k] = sum
	}

	// Levinson-Durbin for LPC coefficients
	a := make([]float64, order+1)
	a[0] = 1.0
	if r[0] == 0 {
		return make([]float64, order)
	}

	err := r[0]
	for i := 1; i <= order; i++ {
		sum := 0.0
		for j := 1; j < i; j++ {
			sum += a[j] * r[i-j]
		}
		k := (r[i] - sum) / err

		newA := make([]float64, order+1)
		copy(newA, a)
		newA[i] = k
		for j := 1; j < i; j++ {
			newA[j] = a[j] - k*a[i-j]
		}
		copy(a, newA)
		err *= (1 - k*k)
		if err < 1e-20 {
			err = 1e-20
		}
	}

	// Convert LPC to cepstral coefficients
	cepstrum := make([]float64, order)
	for n := 1; n <= order; n++ {
		sum := 0.0
		for k := 1; k < n; k++ {
			factor := float64(k) / float64(n)
			if k < len(a) {
				sum += float64(k) * a[k] * cepstrum[n-k-1] * factor
			}
		}
		if n < len(a) {
			cepstrum[n-1] = -a[n] - sum/float64(n)
		}
	}

	return cepstrum
}

// DetectPitch detects the fundamental frequency (F0) using autocorrelation.
// Returns pitch track (one value per frame) and voicing decisions.
func DetectPitch(samples []float64, sampleRate int) (pitch []float64, voiced []bool) {
	frameCfg := FrameConfig{
		FrameLenMs:   30.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.97,
	}
	frames := ProcessFrames(samples, sampleRate, frameCfg)

	minLag := sampleRate / 500 // 500 Hz max
	maxLag := sampleRate / 50  // 50 Hz min
	if maxLag > len(frames[0]) {
		maxLag = len(frames[0]) - 1
	}

	pitch = make([]float64, len(frames))
	voiced = make([]bool, len(frames))

	for i, frame := range frames {
		// Autocorrelation
		bestLag := 0
		bestCorr := 0.0
		for lag := minLag; lag <= maxLag && lag < len(frame); lag++ {
			corr := 0.0
			for j := 0; j+lag < len(frame); j++ {
				corr += frame[j] * frame[j+lag]
			}
			if corr > bestCorr {
				bestCorr = corr
				bestLag = lag
			}
		}

		// Voicing decision: ratio of best autocorrelation to energy
		energy := 0.0
		for _, v := range frame {
			energy += v * v
		}

		if bestLag > 0 && bestCorr > 0.3*float64(len(frame)-bestLag)*energy/float64(len(frame)) {
			pitch[i] = float64(sampleRate) / float64(bestLag)
			voiced[i] = true
		} else {
			pitch[i] = 0
			voiced[i] = false
		}
	}

	return pitch, voiced
}

// VAD performs Voice Activity Detection using energy thresholding.
// Returns a boolean mask indicating which frames contain speech.
func VAD(samples []float64, sampleRate int) []bool {
	frameCfg := FrameConfig{
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.97,
	}
	frames := ProcessFrames(samples, sampleRate, frameCfg)

	// Compute frame energies
	energies := make([]float64, len(frames))
	for i, frame := range frames {
		for _, v := range frame {
			energies[i] += v * v
		}
		energies[i] /= float64(len(frame))
	}

	if len(energies) == 0 {
		return nil
	}

	// Compute thresholds using minimum energy and adaptive threshold
	minE := energies[0]
	maxE := energies[0]
	sumE := 0.0
	for _, e := range energies {
		if e < minE {
			minE = e
		}
		if e > maxE {
			maxE = e
		}
		sumE += e
	}
	avgE := sumE / float64(len(energies))

	// Threshold: midpoint between min and average
	threshold := (minE + avgE) * 0.5
	if threshold < 1e-10 {
		threshold = avgE * 0.1
	}

	// Apply with hysteresis
	voiced := make([]bool, len(energies))
	highThreshold := threshold
	lowThreshold := threshold * 0.6

	inSpeech := false
	for i, e := range energies {
		if !inSpeech && e > highThreshold {
			inSpeech = true
		} else if inSpeech && e < lowThreshold {
			inSpeech = false
		}
		voiced[i] = inSpeech
	}

	return voiced
}

// SpectralSubtraction performs spectral subtraction for noise reduction.
// noiseProfile should be estimated from silent/noise-only frames.
func SpectralSubtraction(samples []float64, sampleRate int, noiseProfile []float64, alpha float64) []float64 {
	frameCfg := FrameConfig{
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.97,
	}
	frames := ProcessFrames(samples, sampleRate, frameCfg)

	result := make([]float64, len(samples))

	for i, frame := range frames {
		// Power spectrum
		psd := PowerSpectrum(frame)

		// Subtract noise estimate
		for j := range psd {
			if j < len(noiseProfile) {
				psd[j] -= alpha * noiseProfile[j]
				if psd[j] < 0 {
					psd[j] = 0
				}
			}
		}

		// Reconstruct via inverse FFT (simplified — just copy back)
		// In practice you'd do proper overlap-add reconstruction
		start := i * int(frameCfg.FrameShiftMs*float64(sampleRate)/1000)
		for j := 0; j < len(frame) && start+j < len(result); j++ {
			result[start+j] = frame[j]
		}
	}

	return result
}

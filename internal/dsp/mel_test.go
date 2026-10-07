package dsp

import (
	"math"
	"testing"
)

func TestHzToMel(t *testing.T) {
	tests := []struct {
		hz, expected float64
	}{
		{0, 0},
		{700, 2595 * math.Log10(2)}, // ~779.76
		{1000, 2595 * math.Log10(1+1000.0/700.0)},
	}

	for _, tt := range tests {
		result := HzToMel(tt.hz)
		if math.Abs(result-tt.expected) > 0.01 {
			t.Errorf("HzToMel(%f): got %f, want %f", tt.hz, result, tt.expected)
		}
	}
}

func TestMelToHz(t *testing.T) {
	tests := []struct {
		mel, expected float64
	}{
		{0, 0},
		{2595 * math.Log10(2), 700},
	}

	for _, tt := range tests {
		result := MelToHz(tt.mel)
		if math.Abs(result-tt.expected) > 0.01 {
			t.Errorf("MelToHz(%f): got %f, want %f", tt.mel, result, tt.expected)
		}
	}
}

func TestHzToMel_MelToHz_RoundTrip(t *testing.T) {
	freqs := []float64{0, 100, 300, 700, 1000, 2000, 4000, 8000}
	for _, hz := range freqs {
		mel := HzToMel(hz)
		recovered := MelToHz(mel)
		if math.Abs(hz-recovered) > 0.01 {
			t.Errorf("round-trip failed for %f Hz: mel=%f, recovered=%f", hz, mel, recovered)
		}
	}
}

func TestMelFilterBank_Basic(t *testing.T) {
	numFilters := 26
	fftSize := 512
	sampleRate := 16000

	filters := MelFilterBank(numFilters, fftSize, sampleRate, 0, 8000)

	if len(filters) != numFilters {
		t.Fatalf("expected %d filters, got %d", numFilters, len(filters))
	}

	fftBins := fftSize/2 + 1
	for i, filter := range filters {
		if len(filter) != fftBins {
			t.Errorf("filter %d: expected length %d, got %d", i, fftBins, len(filter))
		}
	}
}

func TestMelFilterBank_NonNegative(t *testing.T) {
	filters := MelFilterBank(26, 512, 16000, 0, 8000)
	for i, filter := range filters {
		for j, v := range filter {
			if v < -1e-10 {
				t.Errorf("filter[%d][%d] = %f, should be >= 0", i, j, v)
			}
		}
	}
}

func TestMelFilterBank_PeakAtCenter(t *testing.T) {
	filters := MelFilterBank(26, 512, 16000, 0, 8000)
	for i, filter := range filters {
		// Find peak
		maxVal := 0.0
		maxIdx := 0
		for j, v := range filter {
			if v > maxVal {
				maxVal = v
				maxIdx = j
			}
		}
		// Peak should be at most 1.0 (triangular filter)
		if maxVal > 1.0+1e-10 {
			t.Errorf("filter %d: peak = %f, expected <= 1.0", i, maxVal)
		}
		// Peak should not be at the edges (unless filter is very narrow)
		if i > 0 && i < len(filters)-1 {
			if maxIdx == 0 || maxIdx == len(filter)-1 {
				t.Errorf("filter %d: peak at edge (idx=%d)", i, maxIdx)
			}
		}
		_ = maxIdx // suppress unused warning
	}
}

func TestApplyMelFilterBank(t *testing.T) {
	// Create a simple power spectrum
	psd := make([]float64, 257) // 512/2+1
	for i := range psd {
		psd[i] = float64(i) + 1.0 // non-zero
	}

	filters := MelFilterBank(26, 512, 16000, 0, 8000)
	energies := ApplyMelFilterBank(psd, filters)

	if len(energies) != 26 {
		t.Fatalf("expected 26 energies, got %d", len(energies))
	}

	// All energies should be finite
	for i, v := range energies {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("energy[%d] = %f, not finite", i, v)
		}
	}
}

func TestApplyMelFilterBank_ZeroSpectrum(t *testing.T) {
	psd := make([]float64, 257)
	filters := MelFilterBank(26, 512, 16000, 0, 8000)
	energies := ApplyMelFilterBank(psd, filters)

	// Should return log floor values (-20.0)
	for i, v := range energies {
		if v != -20.0 {
			t.Errorf("energy[%d] = %f, expected -20.0 for zero input", i, v)
		}
	}
}

func TestApplyMelFilterBank_MismatchedLengths(t *testing.T) {
	// Power spectrum shorter than filter
	psd := []float64{1.0, 2.0, 3.0}
	filter := []float64{0.5, 0.5, 0.5, 0.5, 0.5}
	filters := [][]float64{filter}

	// Should not panic
	energies := ApplyMelFilterBank(psd, filters)
	if len(energies) != 1 {
		t.Fatalf("expected 1 energy, got %d", len(energies))
	}
}

package dsp

import (
	"math"
	"math/cmplx"
	"testing"
)

func TestFFT_Empty(t *testing.T) {
	result := FFT([]float64{})
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestFFT_SingleElement(t *testing.T) {
	result := FFT([]float64{5.0})
	if len(result) != 1 {
		t.Fatalf("expected 1, got %d", len(result))
	}
	if cmplx.Abs(result[0]-complex(5, 0)) > 1e-10 {
		t.Errorf("expected (5+0i), got %v", result[0])
	}
}

func TestFFT_TwoElements(t *testing.T) {
	result := FFT([]float64{1.0, 2.0})
	if len(result) != 2 {
		t.Fatalf("expected 2, got %d", len(result))
	}
	// X[0] = 1+2 = 3
	// X[1] = 1-2 = -1
	if cmplx.Abs(result[0]-complex(3, 0)) > 1e-10 {
		t.Errorf("X[0]: expected 3, got %v", result[0])
	}
	if cmplx.Abs(result[1]-complex(-1, 0)) > 1e-10 {
		t.Errorf("X[1]: expected -1, got %v", result[1])
	}
}

func TestFFT_FourElements(t *testing.T) {
	// DFT of [1, 0, 1, 0] should be [2, 0, 2, 0]
	result := FFT([]float64{1, 0, 1, 0})
	expected := []complex128{complex(2, 0), complex(0, 0), complex(2, 0), complex(0, 0)}
	for i, v := range result {
		if cmplx.Abs(v-expected[i]) > 1e-10 {
			t.Errorf("X[%d]: got %v, want %v", i, v, expected[i])
		}
	}
}

func TestFFT_PanicsOnNonPowerOf2(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for non-power-of-2 input")
		}
	}()
	FFT([]float64{1, 2, 3}) // length 3 is not power of 2
}

func TestFFT_Correctness(t *testing.T) {
	// Test with a known signal: sinusoid at frequency k
	n := 8
	k := 1
	signal := make([]float64, n)
	for i := 0; i < n; i++ {
		signal[i] = math.Sin(2 * math.Pi * float64(k) * float64(i) / float64(n))
	}

	result := FFT(signal)
	// For a pure sinusoid at frequency k, |X[k]| should be large
	// and |X[n-k]| should be equally large (conjugate symmetry)
	ampK := cmplx.Abs(result[k])
	ampNk := cmplx.Abs(result[n-k])

	if ampK < 3.0 {
		t.Errorf("expected |X[%d]| > 3, got %f", k, ampK)
	}
	if math.Abs(ampK-ampNk) > 0.01 {
		t.Errorf("|X[%d]| and |X[%d]| should be equal: %f vs %f", k, n-k, ampK, ampNk)
	}
}

func TestNextPowerOf2(t *testing.T) {
	tests := []struct {
		input, expected int
	}{
		{0, 1},
		{1, 1},
		{2, 2},
		{3, 4},
		{5, 8},
		{8, 8},
		{15, 16},
		{16, 16},
		{100, 128},
		{1000, 1024},
		{-1, 1},
	}

	for _, tt := range tests {
		result := NextPowerOf2(tt.input)
		if result != tt.expected {
			t.Errorf("NextPowerOf2(%d): got %d, want %d", tt.input, result, tt.expected)
		}
	}
}

func TestPowerSpectrum(t *testing.T) {
	// Create a simple signal
	samples := make([]float64, 64)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 5 * float64(i) / 64)
	}

	psd := PowerSpectrum(samples)
	if len(psd) == 0 {
		t.Fatal("expected non-empty power spectrum")
	}

	// All values should be non-negative
	for i, v := range psd {
		if v < 0 {
			t.Errorf("psd[%d] = %f, should be >= 0", i, v)
		}
	}
}

func TestPowerSpectrum_Empty(t *testing.T) {
	psd := PowerSpectrum([]float64{})
	if len(psd) != 1 { // NextPowerOf2(0) = 1, so PSD has 1 element
		t.Errorf("expected 1 element for empty input, got %d", len(psd))
	}
}

func TestDFT_vs_FFT(t *testing.T) {
	// DFT[k] should match FFT[k] for each k
	signal := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	fftResult := FFT(signal)

	for k := 0; k < len(signal); k++ {
		dftResult := DFT(signal, k)
		if cmplx.Abs(fftResult[k]-dftResult) > 1e-10 {
			t.Errorf("k=%d: FFT=%v, DFT=%v", k, fftResult[k], dftResult)
		}
	}
}

func TestIDFT_RoundTrip(t *testing.T) {
	// Create a real signal, FFT it, then IDFT should recover it
	original := []float64{1, 2, 3, 4, 5, 6, 7, 8}

	fftResult := FFT(original)
	recovered := IDFT(fftResult)

	for i := range original {
		if math.Abs(original[i]-recovered[i]) > 1e-10 {
			t.Errorf("round-trip failed at %d: got %f, want %f", i, recovered[i], original[i])
		}
	}
}

func TestLogEnergy(t *testing.T) {
	tests := []struct {
		name     string
		power    []float64
		expected float64
	}{
		{"empty", []float64{}, -20.0},
		{"zero", []float64{0, 0, 0}, -20.0},
		{"tiny", []float64{1e-25, 1e-25}, -20.0},
		{"normal", []float64{1, 1, 1}, math.Log10(3)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := LogEnergy(tt.power)
			if math.Abs(result-tt.expected) > 1e-6 {
				t.Errorf("LogEnergy: got %f, want %f", result, tt.expected)
			}
		})
	}
}

package dsp

import (
	"math"
	"testing"
)

func TestDefaultMFCCConfig(t *testing.T) {
	cfg := DefaultMFCCConfig(16000)
	if cfg.NumMFCC != 13 {
		t.Errorf("NumMFCC: got %d, want 13", cfg.NumMFCC)
	}
	if cfg.NumFilters != 26 {
		t.Errorf("NumFilters: got %d, want 26", cfg.NumFilters)
	}
	if cfg.FFTSize != 512 {
		t.Errorf("FFTSize: got %d, want 512", cfg.FFTSize)
	}
	if cfg.SampleRate != 16000 {
		t.Errorf("SampleRate: got %d, want 16000", cfg.SampleRate)
	}
	if !cfg.IncludeEnergy {
		t.Error("IncludeEnergy should be true")
	}
}

func TestExtractMFCC_EmptyFrames(t *testing.T) {
	cfg := DefaultMFCCConfig(16000)
	result := ExtractMFCC([][]float64{}, cfg)
	if len(result.Coefficients) != 0 {
		t.Errorf("expected 0 coefficients, got %d", len(result.Coefficients))
	}
}

func TestExtractMFCC_SingleFrame(t *testing.T) {
	cfg := DefaultMFCCConfig(16000)
	frame := make([]float64, 400)
	for i := range frame {
		frame[i] = math.Sin(2*math.Pi*440*float64(i)/16000) * 0.5
	}

	result := ExtractMFCC([][]float64{frame}, cfg)
	if len(result.Coefficients) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(result.Coefficients))
	}

	// Should have NumMFCC + 1 (energy) coefficients
	expectedCoeffs := cfg.NumMFCC + 1
	if len(result.Coefficients[0]) != expectedCoeffs {
		t.Errorf("expected %d coefficients per frame, got %d", expectedCoeffs, len(result.Coefficients[0]))
	}

	// All coefficients should be finite
	for j, v := range result.Coefficients[0] {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("coefficient[%d] = %f, not finite", j, v)
		}
	}
}

func TestExtractMFCC_MultipleFrames(t *testing.T) {
	cfg := DefaultMFCCConfig(16000)
	numFrames := 10
	frames := make([][]float64, numFrames)
	for i := range frames {
		frames[i] = make([]float64, 400)
		for j := range frames[i] {
			frames[i][j] = math.Sin(2*math.Pi*440*float64(j)/16000) * 0.5
		}
	}

	result := ExtractMFCC(frames, cfg)
	if len(result.Coefficients) != numFrames {
		t.Errorf("expected %d frames, got %d", numFrames, len(result.Coefficients))
	}

	// Deltas should have same number of frames
	if len(result.Deltas) != numFrames {
		t.Errorf("expected %d delta frames, got %d", numFrames, len(result.Deltas))
	}

	// Delta-deltas should have same number of frames
	if len(result.DeltaDeltas) != numFrames {
		t.Errorf("expected %d delta-delta frames, got %d", numFrames, len(result.DeltaDeltas))
	}
}

func TestExtractMFCC_NoEnergy(t *testing.T) {
	cfg := DefaultMFCCConfig(16000)
	cfg.IncludeEnergy = false

	frame := make([]float64, 400)
	result := ExtractMFCC([][]float64{frame}, cfg)

	// Without energy, should have exactly NumMFCC coefficients
	if len(result.Coefficients[0]) != cfg.NumMFCC {
		t.Errorf("expected %d coefficients without energy, got %d", cfg.NumMFCC, len(result.Coefficients[0]))
	}
}

func TestExtractMFCCFromRaw(t *testing.T) {
	// Generate 1 second of audio at 16kHz
	sampleRate := 16000
	samples := make([]float64, sampleRate)
	for i := range samples {
		samples[i] = math.Sin(2*math.Pi*440*float64(i)/float64(sampleRate)) * 0.5
	}

	result := ExtractMFCCFromRaw(samples, sampleRate)
	if len(result.Coefficients) == 0 {
		t.Fatal("expected at least 1 frame")
	}

	// Check dimensions
	expectedCoeffs := 14 // 13 MFCC + 1 energy
	for i, frame := range result.Coefficients {
		if len(frame) != expectedCoeffs {
			t.Errorf("frame %d: expected %d coefficients, got %d", i, expectedCoeffs, len(frame))
		}
	}
}

func TestExtractMFCCFromRaw_VeryShort(t *testing.T) {
	// Very short signal — should not panic
	samples := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	result := ExtractMFCCFromRaw(samples, 16000)
	if len(result.Coefficients) == 0 {
		t.Fatal("expected at least 1 frame")
	}
}

func TestExtractMFCCFromRaw_Empty(t *testing.T) {
	result := ExtractMFCCFromRaw([]float64{}, 16000)
	if len(result.Coefficients) != 0 {
		t.Errorf("expected 0 frames for empty input, got %d", len(result.Coefficients))
	}
}

func TestConcatenateFeatures(t *testing.T) {
	coeffs := [][]float64{{1, 2, 3}, {4, 5, 6}}
	deltas := [][]float64{{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}}
	deltaDeltas := [][]float64{{0.01, 0.02, 0.03}, {0.04, 0.05, 0.06}}

	result := MFCCResult{
		Coefficients: coeffs,
		Deltas:       deltas,
		DeltaDeltas:  deltaDeltas,
	}

	features := ConcatenateFeatures(result)
	if len(features) != 2 {
		t.Fatalf("expected 2 frames, got %d", len(features))
	}
	if len(features[0]) != 9 {
		t.Errorf("expected 9 features per frame, got %d", len(features[0]))
	}

	// Check values
	if features[0][0] != 1.0 || features[0][3] != 0.1 || features[0][6] != 0.01 {
		t.Errorf("concatenation values incorrect: %v", features[0])
	}
}

func TestConcatenateFeatures_Empty(t *testing.T) {
	result := MFCCResult{
		Coefficients: [][]float64{},
	}
	features := ConcatenateFeatures(result)
	if features != nil {
		t.Errorf("expected nil for empty input, got %v", features)
	}
}

func TestComputeDeltas_SingleFrame(t *testing.T) {
	features := [][]float64{{1.0, 2.0, 3.0}}
	deltas := computeDeltas(features)
	if len(deltas) != 1 {
		t.Fatalf("expected 1 delta frame, got %d", len(deltas))
	}
	// Single frame delta should be zero
	for i, v := range deltas[0] {
		if v != 0 {
			t.Errorf("delta[%d] = %f, expected 0 for single frame", i, v)
		}
	}
}

func TestComputeDeltas_TwoFrames(t *testing.T) {
	features := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	deltas := computeDeltas(features)
	if len(deltas) != 2 {
		t.Fatalf("expected 2 delta frames, got %d", len(deltas))
	}
	// First frame: features[1] - features[0]
	if deltas[0][0] != 2.0 || deltas[0][1] != 2.0 {
		t.Errorf("first frame delta incorrect: %v", deltas[0])
	}
	// Last frame: features[1] - features[0] (same as first for 2 frames)
	if deltas[1][0] != 2.0 || deltas[1][1] != 2.0 {
		t.Errorf("last frame delta incorrect: %v", deltas[1])
	}
}

func TestMakeDCTMatrix(t *testing.T) {
	matrix := makeDCTMatrix(4, 3)
	if len(matrix) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(matrix))
	}
	for i, row := range matrix {
		if len(row) != 4 {
			t.Errorf("row %d: expected 4 cols, got %d", i, len(row))
		}
	}
}

func TestApplyDCT(t *testing.T) {
	// DCT of a constant signal should have non-zero 0th coefficient
	x := []float64{1.0, 1.0, 1.0, 1.0}
	matrix := makeDCTMatrix(4, 4)
	result := applyDCT(x, matrix)

	// 0th DCT coefficient should be non-zero (sum of input)
	if math.Abs(result[0]) < 1e-10 {
		t.Error("0th DCT coefficient should be non-zero for constant input")
	}
}

func TestProcessFrames_Empty(t *testing.T) {
	cfg := FrameConfig{FrameLenMs: 25, FrameShiftMs: 10, PreEmphCoeff: 0.97}
	result := ProcessFrames([]float64{}, 16000, cfg)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestProcessFrames_ShortSignal(t *testing.T) {
	cfg := FrameConfig{FrameLenMs: 25, FrameShiftMs: 10, PreEmphCoeff: 0.97}
	samples := make([]float64, 50) // very short
	result := ProcessFrames(samples, 16000, cfg)
	if len(result) != 1 {
		t.Fatalf("expected 1 frame for short signal, got %d", len(result))
	}
	if len(result[0]) != 400 {
		t.Errorf("expected frame length 400, got %d", len(result[0]))
	}
}

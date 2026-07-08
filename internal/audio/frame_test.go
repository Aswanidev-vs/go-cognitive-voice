package audio

import (
	"math"
	"testing"
)

func TestPreEmphasis(t *testing.T) {
	tests := []struct {
		name     string
		samples  []float64
		coeff    float64
		expected []float64
	}{
		{
			name:     "empty input",
			samples:  []float64{},
			coeff:    0.97,
			expected: nil,
		},
		{
			name:     "single sample",
			samples:  []float64{1.0},
			coeff:    0.97,
			expected: []float64{1.0},
		},
		{
			name:     "two samples",
			samples:  []float64{1.0, 0.5},
			coeff:    0.97,
			expected: []float64{1.0, 0.5 - 0.97*1.0},
		},
		{
			name:     "zero coefficient",
			samples:  []float64{1.0, 2.0, 3.0},
			coeff:    0.0,
			expected: []float64{1.0, 2.0, 3.0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := PreEmphasis(tt.samples, tt.coeff)
			if tt.expected == nil {
				if result != nil {
					t.Errorf("expected nil, got %v", result)
				}
				return
			}
			if len(result) != len(tt.expected) {
				t.Fatalf("length mismatch: got %d, want %d", len(result), len(tt.expected))
			}
			for i := range result {
				if math.Abs(result[i]-tt.expected[i]) > 1e-10 {
					t.Errorf("index %d: got %f, want %f", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestFrame_EmptyInput(t *testing.T) {
	cfg := DefaultFrameConfig()
	result := Frame([]float64{}, 16000, cfg)
	if result != nil {
		t.Errorf("expected nil for empty input, got %v", result)
	}
}

func TestFrame_ShortSignal(t *testing.T) {
	cfg := DefaultFrameConfig()
	// Signal shorter than one frame (25ms @ 16000Hz = 400 samples)
	samples := make([]float64, 100)
	for i := range samples {
		samples[i] = float64(i)
	}
	result := Frame(samples, 16000, cfg)
	if len(result) != 1 {
		t.Fatalf("expected 1 frame for short signal, got %d", len(result))
	}
	if len(result[0]) != 400 {
		t.Errorf("expected frame length 400, got %d", len(result[0]))
	}
}

func TestFrame_SingleSample(t *testing.T) {
	cfg := DefaultFrameConfig()
	result := Frame([]float64{0.5}, 16000, cfg)
	if len(result) != 1 {
		t.Fatalf("expected 1 frame, got %d", len(result))
	}
}

func TestFrame_NormalSignal(t *testing.T) {
	cfg := DefaultFrameConfig()
	// 1 second of audio at 16000Hz
	samples := make([]float64, 16000)
	for i := range samples {
		samples[i] = math.Sin(2 * math.Pi * 440 * float64(i) / 16000)
	}
	result := Frame(samples, 16000, cfg)
	if len(result) == 0 {
		t.Fatal("expected at least 1 frame")
	}
	// Each frame should be 400 samples (25ms @ 16kHz)
	for i, frame := range result {
		if len(frame) != 400 {
			t.Errorf("frame %d: expected length 400, got %d", i, len(frame))
		}
	}
}

func TestHammingWindow(t *testing.T) {
	tests := []struct {
		name  string
		input []float64
	}{
		{"empty", []float64{}},
		{"single element", []float64{5.0}},
		{"two elements", []float64{1.0, 2.0}},
		{"normal frame", make([]float64, 400)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := make([]float64, len(tt.input))
			copy(original, tt.input)
			HammingWindow(tt.input)
			// Should not panic and should modify in-place
			if len(tt.input) > 1 {
				// First and last elements should be scaled differently
				// Hamming at edges: 0.54 - 0.46*cos(0) = 0.08
				if math.Abs(tt.input[0]) > 1e-10 && tt.input[0] == original[0] {
					t.Error("HammingWindow should modify values")
				}
			}
		})
	}
}

func TestHammingWindow_NoPanicOnLength1(t *testing.T) {
	frame := []float64{42.0}
	HammingWindow(frame)
	if frame[0] != 42.0 {
		t.Error("single element should not be modified")
	}
}

func TestHanningWindow_NoPanicOnLength1(t *testing.T) {
	frame := []float64{42.0}
	HanningWindow(frame)
	if frame[0] != 42.0 {
		t.Error("single element should not be modified")
	}
}

func TestProcessFrames_Empty(t *testing.T) {
	cfg := DefaultFrameConfig()
	result := ProcessFrames([]float64{}, 16000, cfg)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

func TestProcessFrames_ShortSignal(t *testing.T) {
	cfg := DefaultFrameConfig()
	samples := make([]float64, 100) // very short
	result := ProcessFrames(samples, 16000, cfg)
	if len(result) != 1 {
		t.Fatalf("expected 1 frame for short signal, got %d", len(result))
	}
}

func TestProcessFrames_Windowing(t *testing.T) {
	cfg := FrameConfig{
		FrameLenMs:   25.0,
		FrameShiftMs: 10.0,
		PreEmphCoeff: 0.0, // no pre-emphasis for clarity
	}
	// Create a signal with a single impulse
	samples := make([]float64, 16000)
	samples[0] = 1.0
	result := ProcessFrames(samples, 16000, cfg)
	if len(result) == 0 {
		t.Fatal("expected frames")
	}
	// First frame should have the impulse modified by Hamming window
	for _, val := range result[0] {
		if val > 1.0 || val < -1.0 {
			// Hamming window should keep values bounded
			t.Errorf("value out of Hamming range: %f", val)
		}
	}
}

func TestDefaultFrameConfig(t *testing.T) {
	cfg := DefaultFrameConfig()
	if cfg.FrameLenMs != 25.0 {
		t.Errorf("FrameLenMs: got %f, want 25.0", cfg.FrameLenMs)
	}
	if cfg.FrameShiftMs != 10.0 {
		t.Errorf("FrameShiftMs: got %f, want 10.0", cfg.FrameShiftMs)
	}
	if cfg.PreEmphCoeff != 0.97 {
		t.Errorf("PreEmphCoeff: got %f, want 0.97", cfg.PreEmphCoeff)
	}
}

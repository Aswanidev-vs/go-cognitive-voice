package matcher

import (
	"math"
	"testing"
)

func TestDTW_EmptySequences(t *testing.T) {
	cfg := DefaultDTWConfig()
	result := DTW(nil, [][]float64{{1}}, cfg)
	if !math.IsInf(result.Distance, 1) {
		t.Errorf("expected +Inf for empty seq1, got %f", result.Distance)
	}

	result = DTW([][]float64{{1}}, nil, cfg)
	if !math.IsInf(result.Distance, 1) {
		t.Errorf("expected +Inf for empty seq2, got %f", result.Distance)
	}

	result = DTW(nil, nil, cfg)
	if !math.IsInf(result.Distance, 1) {
		t.Errorf("expected +Inf for both empty, got %f", result.Distance)
	}
}

func TestDTW_IdenticalSequences(t *testing.T) {
	cfg := DefaultDTWConfig()
	seq := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	result := DTW(seq, seq, cfg)

	// Identical sequences should have distance 0 (or very close)
	if result.Distance > 1e-10 {
		t.Errorf("identical sequences: distance = %f, expected ~0", result.Distance)
	}
}

func TestDTW_SingleElement(t *testing.T) {
	cfg := DefaultDTWConfig()
	a := [][]float64{{1, 2}}
	b := [][]float64{{1, 2}}
	result := DTW(a, b, cfg)
	if result.Distance > 1e-10 {
		t.Errorf("single identical elements: distance = %f", result.Distance)
	}

	c := [][]float64{{5, 6}}
	result = DTW(a, c, cfg)
	if result.Distance <= 0 {
		t.Error("different single elements should have positive distance")
	}
}

func TestDTW_Symmetric(t *testing.T) {
	cfg := DefaultDTWConfig()
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}

	resultAB := DTW(a, b, cfg)
	resultBA := DTW(b, a, cfg)

	if math.Abs(resultAB.Distance-resultBA.Distance) > 1e-10 {
		t.Errorf("DTW not symmetric: AB=%f, BA=%f", resultAB.Distance, resultBA.Distance)
	}
}

func TestDTW_DifferentLengths(t *testing.T) {
	cfg := DefaultDTWConfig()
	a := [][]float64{{1}, {2}, {3}, {4}, {5}}
	b := [][]float64{{1}, {3}, {5}}

	result := DTW(a, b, cfg)
	if result.Distance < 0 {
		t.Errorf("distance should be non-negative, got %f", result.Distance)
	}
}

func TestDTW_NormalizedMode(t *testing.T) {
	cfg := DTWConfig{Mode: DTWModeNormalized}
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}

	result := DTW(a, b, cfg)
	if result.Distance <= 0 {
		t.Error("normalized distance should be positive for different sequences")
	}
}

func TestDTW_PaddedMode(t *testing.T) {
	cfg := DTWConfig{Mode: DTWModePadded}
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{5, 6}, {7, 8}}

	result := DTW(a, b, cfg)
	if result.Distance <= 0 {
		t.Error("padded distance should be positive")
	}
}

func TestDTW_SakoeChibaWindow(t *testing.T) {
	cfg := DTWConfig{Mode: DTWModeNormalized, Window: 1}
	a := [][]float64{{1}, {2}, {3}, {4}, {5}}
	b := [][]float64{{1}, {2}, {3}, {4}, {5}}

	result := DTW(a, b, cfg)
	if result.Distance > 1e-10 {
		t.Errorf("identical sequences with window: distance = %f", result.Distance)
	}
}

func TestDTW_LogDistance(t *testing.T) {
	cfg := DTWConfig{Mode: DTWModeNormalized, UseLog: true}
	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{1, 2}, {3, 4}}

	result := DTW(a, b, cfg)
	if result.Distance > 1e-10 {
		t.Errorf("identical with log: distance = %f", result.Distance)
	}
}

func TestDTWWithPath(t *testing.T) {
	cfg := DefaultDTWConfig()
	a := [][]float64{{1, 2}, {3, 4}, {5, 6}}
	b := [][]float64{{1, 2}, {3, 4}, {5, 6}}

	result := DTWWithPath(a, b, cfg)
	if len(result.Path) == 0 {
		t.Error("expected non-empty path")
	}

	// Path should start near (0,0) and end near (n-1,m-1)
	first := result.Path[0]
	last := result.Path[len(result.Path)-1]

	if first[0] != 0 || first[1] != 0 {
		t.Errorf("path should start at (0,0), got (%d,%d)", first[0], first[1])
	}
	if last[0] != 2 || last[1] != 2 {
		t.Errorf("path should end at (2,2), got (%d,%d)", last[0], last[1])
	}
}

func TestDTWWithPath_Empty(t *testing.T) {
	cfg := DefaultDTWConfig()
	result := DTWWithPath(nil, [][]float64{{1}}, cfg)
	if len(result.Path) != 0 {
		t.Error("expected empty path for empty input")
	}
}

func TestDTW_MonotonicPath(t *testing.T) {
	cfg := DefaultDTWConfig()
	a := [][]float64{{1}, {2}, {3}, {4}}
	b := [][]float64{{1.5}, {2.5}, {3.5}}

	result := DTWWithPath(a, b, cfg)
	// Path should be monotonically increasing in both dimensions
	for i := 1; i < len(result.Path); i++ {
		if result.Path[i][0] < result.Path[i-1][0] || result.Path[i][1] < result.Path[i-1][1] {
			t.Errorf("path not monotonic at step %d: (%d,%d) -> (%d,%d)",
				i, result.Path[i-1][0], result.Path[i-1][1],
				result.Path[i][0], result.Path[i][1])
		}
	}
}

func TestFeatureDistance(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	expected := math.Sqrt(27) // sqrt(9+9+9)

	result := featureDistance(a, b, false)
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("featureDistance: got %f, want %f", result, expected)
	}
}

func TestFeatureDistance_Empty(t *testing.T) {
	result := featureDistance([]float64{}, []float64{}, false)
	if result != 0 {
		t.Errorf("expected 0 for empty vectors, got %f", result)
	}
}

func TestFeatureDistance_DifferentLengths(t *testing.T) {
	a := []float64{1, 2, 3, 4, 5}
	b := []float64{1, 2}
	// Should only use first 2 elements: sqrt((1-1)^2 + (2-2)^2) = 0
	result := featureDistance(a, b, false)
	if result != 0 {
		t.Errorf("expected 0, got %f", result)
	}
}

func TestMin3(t *testing.T) {
	tests := []struct {
		a, b, c, expected float64
	}{
		{1, 2, 3, 1},
		{3, 2, 1, 1},
		{2, 1, 3, 1},
		{1, 1, 1, 1},
		{-1, 0, 1, -1},
	}

	for _, tt := range tests {
		result := min3(tt.a, tt.b, tt.c)
		if result != tt.expected {
			t.Errorf("min3(%f,%f,%f): got %f, want %f", tt.a, tt.b, tt.c, result, tt.expected)
		}
	}
}

func TestDefaultDTWConfig(t *testing.T) {
	cfg := DefaultDTWConfig()
	if cfg.Mode != DTWModeNormalized {
		t.Errorf("Mode: got %d, want DTWModeNormalized", cfg.Mode)
	}
	if cfg.Window != 0 {
		t.Errorf("Window: got %d, want 0", cfg.Window)
	}
	if cfg.UseLog {
		t.Error("UseLog should be false by default")
	}
}

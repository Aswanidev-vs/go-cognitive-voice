package matcher

import (
	"math"
	"testing"
)

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float64
		expected float64
		epsilon  float64
	}{
		{"identical", []float64{1, 2, 3}, []float64{1, 2, 3}, 1.0, 1e-10},
		{"opposite", []float64{1, 0}, []float64{-1, 0}, -1.0, 1e-10},
		{"orthogonal", []float64{1, 0}, []float64{0, 1}, 0.0, 1e-10},
		{"same direction different magnitude", []float64{2, 4}, []float64{1, 2}, 1.0, 1e-10},
		{"empty a", []float64{}, []float64{1, 2}, 0.0, 1e-10},
		{"empty b", []float64{1, 2}, []float64{}, 0.0, 1e-10},
		{"both empty", []float64{}, []float64{}, 0.0, 1e-10},
		{"different lengths", []float64{1, 2}, []float64{1, 2, 3}, 0.0, 1e-10},
		{"zero vector a", []float64{0, 0}, []float64{1, 2}, 0.0, 1e-10},
		{"zero vector b", []float64{1, 2}, []float64{0, 0}, 0.0, 1e-10},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineSimilarity(tt.a, tt.b)
			if math.Abs(result-tt.expected) > tt.epsilon {
				t.Errorf("CosineSimilarity: got %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestCosineDistance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float64
		expected float64
	}{
		{"identical", []float64{1, 2}, []float64{1, 2}, 0.0},
		{"opposite", []float64{1, 0}, []float64{-1, 0}, 2.0},
		{"orthogonal", []float64{1, 0}, []float64{0, 1}, 1.0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := CosineDistance(tt.a, tt.b)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("CosineDistance: got %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestEuclideanDistance(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float64
		expected float64
	}{
		{"identical", []float64{1, 2}, []float64{1, 2}, 0.0},
		{"unit x", []float64{0, 0}, []float64{1, 0}, 1.0},
		{"unit y", []float64{0, 0}, []float64{0, 1}, 1.0},
		{"diagonal", []float64{0, 0}, []float64{3, 4}, 5.0},
		{"different lengths", []float64{1}, []float64{1, 2}, math.Inf(1)},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := EuclideanDistance(tt.a, tt.b)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("EuclideanDistance: got %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestMahalanobisDistance_NilInvCov(t *testing.T) {
	a := []float64{0, 0}
	b := []float64{3, 4}
	result := MahalanobisDistance(a, b, nil, 2)
	expected := 5.0 // should fall back to Euclidean
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestMahalanobisDistance_IdentityInvCov(t *testing.T) {
	a := []float64{0, 0}
	b := []float64{3, 4}
	// Identity inverse covariance => same as Euclidean
	invCov := []float64{1, 0, 0, 1}
	result := MahalanobisDistance(a, b, invCov, 2)
	expected := 5.0
	if math.Abs(result-expected) > 1e-10 {
		t.Errorf("expected %f, got %f", expected, result)
	}
}

func TestMeanVector(t *testing.T) {
	tests := []struct {
		name     string
		vectors  [][]float64
		expected []float64
	}{
		{"empty", nil, nil},
		{"single", [][]float64{{1, 2, 3}}, []float64{1, 2, 3}},
		{"two vectors", [][]float64{{2, 4}, {4, 8}}, []float64{3, 6}},
		{"three vectors", [][]float64{{1, 0}, {0, 1}, {0, 0}}, []float64{1.0 / 3, 1.0 / 3}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := MeanVector(tt.vectors)
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

func TestCovarianceMatrix(t *testing.T) {
	// Two vectors [1,2] and [3,4]
	vectors := [][]float64{{1, 2}, {3, 4}}
	cov := CovarianceMatrix(vectors)

	if len(cov) != 2 {
		t.Fatalf("expected 2x2 covariance, got %dx%d", len(cov), len(cov[0]))
	}

	// Covariance of 2 vectors: each element is (x1-m1)(x2-m2)/(n-1)
	// Mean = [2, 3]
	// (1-2)*(1-2)/1 = 1, (1-2)*(2-3)/1 = 1, etc.
	expected := [][]float64{{2, 2}, {2, 2}}
	for i := range cov {
		for j := range cov[i] {
			if math.Abs(cov[i][j]-expected[i][j]) > 1e-10 {
				t.Errorf("cov[%d][%d]: got %f, want %f", i, j, cov[i][j], expected[i][j])
			}
		}
	}
}

func TestNormalizeVector(t *testing.T) {
	v := []float64{3, 4}
	result := NormalizeVector(v)
	norm := 0.0
	for _, x := range result {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if math.Abs(norm-1.0) > 1e-10 {
		t.Errorf("normalized vector norm: got %f, want 1.0", norm)
	}
}

func TestNormalizeVector_ZeroVector(t *testing.T) {
	v := []float64{0, 0, 0}
	result := NormalizeVector(v)
	for i, x := range result {
		if x != 0 {
			t.Errorf("index %d: got %f, want 0", i, x)
		}
	}
}

func TestStandardize(t *testing.T) {
	vectors := [][]float64{{1, 10}, {2, 20}, {3, 30}}
	result := Standardize(vectors)

	if len(result) != 3 {
		t.Fatalf("expected 3 frames, got %d", len(result))
	}

	// After standardization, mean should be ~0 and stddev ~1
	mean := MeanVector(result)
	for i, m := range mean {
		if math.Abs(m) > 1e-10 {
			t.Errorf("mean[%d] = %f, expected ~0", i, m)
		}
	}
}

func TestStandardize_ConstantFeature(t *testing.T) {
	// Constant feature => stddev=0 => should use stddev=1 (no division)
	vectors := [][]float64{{5, 1}, {5, 2}, {5, 3}}
	result := Standardize(vectors)

	// First feature should remain 5 (mean=5, stddev=1 => (5-5)/1=0)
	for _, v := range result {
		if v[0] != 0 {
			t.Errorf("constant feature should standardize to 0, got %f", v[0])
		}
	}
}

func TestStandardize_Empty(t *testing.T) {
	result := Standardize(nil)
	if result != nil {
		t.Errorf("expected nil, got %v", result)
	}
}

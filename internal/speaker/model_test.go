package speaker

import (
	"math"
	"testing"

	"github.com/Aswanidev-vs/go-cognitive-voice/internal/matcher"
)

func generateCluster(center []float64, n int, noise float64) [][]float64 {
	features := make([][]float64, n)
	for i := range features {
		features[i] = make([]float64, len(center))
		for j, c := range center {
			features[i][j] = c + (math.Sin(float64(i*j+1)) * noise)
		}
	}
	return features
}

func TestNewSpeakerModel(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 39)
	if m.ID != "spk1" {
		t.Errorf("ID: got %s, want spk1", m.ID)
	}
	if m.Name != "Alice" {
		t.Errorf("Name: got %s, want Alice", m.Name)
	}
	if m.Dim != 39 {
		t.Errorf("Dim: got %d, want 39", m.Dim)
	}
}

func TestTrainCentroid(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	m.TrainCentroid(features)

	if m.TrainMethod != "centroid" {
		t.Errorf("TrainMethod: got %s, want centroid", m.TrainMethod)
	}
	if m.NumSamples != 20 {
		t.Errorf("NumSamples: got %d, want 20", m.NumSamples)
	}
	if m.Centroid == nil {
		t.Fatal("Centroid should not be nil")
	}
	if len(m.Centroid) != 3 {
		t.Fatalf("Centroid dim: got %d, want 3", len(m.Centroid))
	}
	// Centroid should be close to the cluster center
	for i, v := range m.Centroid {
		if math.Abs(v-float64(i+1)) > 0.5 {
			t.Errorf("Centroid[%d] = %f, expected near %d", i, v, i+1)
		}
	}
}

func TestTrainCentroid_EmptyFeatures(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	m.TrainCentroid([][]float64{})
	if m.Centroid != nil {
		t.Error("centroid should be nil for empty features")
	}
}

func TestScoreCentroid_SelfMatch(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	m.TrainCentroid(features)

	// Querying with the centroid itself should yield high score
	score := m.Score(m.Centroid)
	if score < 0.9 {
		t.Errorf("self-match score: got %f, want >= 0.9", score)
	}
}

func TestScoreCentroid_DifferentSpeaker(t *testing.T) {
	m1 := NewSpeakerModel("spk1", "Alice", 3)
	m2 := NewSpeakerModel("spk2", "Bob", 3)

	f1 := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	f2 := generateCluster([]float64{10, 20, 30}, 20, 0.1)

	m1.TrainCentroid(f1)
	m2.TrainCentroid(f2)

	// Alice's voice should score higher on Alice than on Bob
	scoreAlice := m1.Score(m1.Centroid)
	scoreBob := m1.Score(m2.Centroid)

	if scoreAlice <= scoreBob {
		t.Errorf("self-score (%f) should be > cross-score (%f)", scoreAlice, scoreBob)
	}
}

func TestTrainDTW(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 10, 0.1)
	m.TrainDTW(features)

	if m.TrainMethod != "dtw" {
		t.Errorf("TrainMethod: got %s, want dtw", m.TrainMethod)
	}
	if len(m.Features) != 10 {
		t.Errorf("Features: got %d, want 10", len(m.Features))
	}
}

func TestScoreSequence_DTW(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 10, 0.1)
	m.TrainDTW(features)

	// Same features should give high score
	score := m.ScoreSequence(features)
	if score < 0.9 {
		t.Errorf("DTW self-match: got %f, want >= 0.9", score)
	}
}

func TestScoreSequence_DifferentSpeaker(t *testing.T) {
	m1 := NewSpeakerModel("spk1", "Alice", 3)
	m2 := NewSpeakerModel("spk2", "Bob", 3)

	f1 := generateCluster([]float64{1, 2, 3}, 10, 0.1)
	f2 := generateCluster([]float64{10, 20, 30}, 10, 0.1)

	m1.TrainDTW(f1)
	m2.TrainDTW(f2)

	scoreAlice := m1.ScoreSequence(f1)
	scoreBob := m1.ScoreSequence(f2)

	if scoreAlice <= scoreBob {
		t.Errorf("DTW self-score (%f) should be > cross-score (%f)", scoreAlice, scoreBob)
	}
}

func TestTrainGMM(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 50, 0.2)
	m.TrainGMM(features, 3, 50, 1e-6)

	if m.TrainMethod != "gmm" {
		t.Errorf("TrainMethod: got %s, want gmm", m.TrainMethod)
	}
	if m.GMM == nil {
		t.Fatal("GMM should not be nil")
	}
	if m.GMM.K != 3 {
		t.Errorf("GMM.K: got %d, want 3", m.GMM.K)
	}
}

func TestTrainGMM_SmallK(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 10, 0.1)
	// k=0 should auto-select k
	m.TrainGMM(features, 0, 50, 1e-6)

	if m.GMM == nil {
		t.Fatal("GMM should not be nil")
	}
	if m.GMM.K < 2 {
		t.Errorf("GMM.K should be >= 2, got %d", m.GMM.K)
	}
}

func TestTrainGMM_KExceedsData(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 5, 0.1)
	// k=10 exceeds data size (5) => should be clamped
	m.TrainGMM(features, 10, 50, 1e-6)

	if m.GMM == nil {
		t.Fatal("GMM should not be nil")
	}
	if m.GMM.K > 5 {
		t.Errorf("GMM.K should be clamped to data size, got %d", m.GMM.K)
	}
}

func TestScoreGMM(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 50, 0.2)
	m.TrainGMM(features, 3, 50, 1e-6)

	// Mean of cluster should score higher than distant point
	mean := matcher.MeanVector(features)
	distant := []float64{100, 200, 300}

	scoreClose := m.Score(mean)
	scoreDistant := m.Score(distant)

	if scoreClose <= scoreDistant {
		t.Errorf("close score (%f) should be > distant score (%f)", scoreClose, scoreDistant)
	}
}

func TestScore_NoMethod(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	// No training
	score := m.Score([]float64{1, 2, 3})
	if score != 0 {
		t.Errorf("expected 0 for untrained model, got %f", score)
	}
}

func TestScoreSequence_NoFeatures(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	// No training, no centroid
	score := m.ScoreSequence([][]float64{{1, 2, 3}})
	if score != 0 {
		t.Errorf("expected 0 for untrained model, got %f", score)
	}
}

func TestInvertMatrix(t *testing.T) {
	// Identity matrix => inverse is identity
	flat := []float64{1, 0, 0, 1}
	inv := invertMatrix(flat, 2)
	if inv == nil {
		t.Fatal("inverse should not be nil for identity")
	}
	for i, v := range inv {
		expected := 0.0
		if i%3 == 0 {
			expected = 1.0
		}
		if math.Abs(v-expected) > 1e-10 {
			t.Errorf("inv[%d] = %f, want %f", i, v, expected)
		}
	}
}

func TestInvertMatrix_Singular(t *testing.T) {
	// Singular matrix (row of zeros)
	flat := []float64{0, 0, 0, 0}
	inv := invertMatrix(flat, 2)
	if inv != nil {
		t.Error("expected nil for singular matrix")
	}
}

func TestInvertMatrix_ZeroDim(t *testing.T) {
	inv := invertMatrix([]float64{}, 0)
	if inv != nil {
		t.Error("expected nil for zero dim")
	}
}

func TestMatrixDeterminant(t *testing.T) {
	tests := []struct {
		name     string
		m        [][]float64
		n        int
		expected float64
	}{
		{"1x1", [][]float64{{5}}, 1, 5},
		{"2x2", [][]float64{{1, 2}, {3, 4}}, 2, -2},
		{"identity3x3", identityMatrix(3), 3, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := matrixDeterminant(tt.m, tt.n)
			if math.Abs(result-tt.expected) > 1e-10 {
				t.Errorf("det: got %f, want %f", result, tt.expected)
			}
		})
	}
}

func TestFlatMatrix(t *testing.T) {
	m := [][]float64{{1, 2}, {3, 4}}
	flat := flatMatrix(m)
	expected := []float64{1, 2, 3, 4}
	for i, v := range flat {
		if v != expected[i] {
			t.Errorf("flat[%d] = %f, want %f", i, v, expected[i])
		}
	}
}

func TestIdentityMatrix(t *testing.T) {
	m := identityMatrix(3)
	if len(m) != 3 {
		t.Fatalf("expected 3 rows, got %d", len(m))
	}
	for i := range m {
		if len(m[i]) != 3 {
			t.Fatalf("row %d: expected 3 cols, got %d", i, len(m[i]))
		}
		for j := range m[i] {
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if m[i][j] != expected {
				t.Errorf("m[%d][%d] = %f, want %f", i, j, m[i][j], expected)
			}
		}
	}
}

func TestLogAdd(t *testing.T) {
	tests := []struct {
		a, b, expected float64
	}{
		{math.Log(2), math.Log(3), math.Log(5)},
		{math.Inf(-1), math.Log(1), math.Log(1)},
		{math.Log(1), math.Inf(-1), math.Log(1)},
		{math.Inf(-1), math.Inf(-1), math.Inf(-1)},
	}

	for _, tt := range tests {
		result := logAdd(tt.a, tt.b)
		if math.Abs(result-tt.expected) > 1e-10 {
			t.Errorf("logAdd(%f, %f): got %f, want %f", tt.a, tt.b, result, tt.expected)
		}
	}
}

func TestGaussianPDF_AtMean(t *testing.T) {
	mean := []float64{0, 0}
	covar := identityMatrix(2)
	x := []float64{0, 0}

	pdf := gaussianPDF(x, mean, covar)
	// At mean, PDF should be 1/(2*pi) for identity covariance
	expected := 1.0 / (2 * math.Pi)
	if math.Abs(pdf-expected) > 1e-6 {
		t.Errorf("PDF at mean: got %f, want %f", pdf, expected)
	}
}

func TestGaussianPDF_DimMismatch(t *testing.T) {
	x := []float64{1, 2}
	mean := []float64{1} // different dim
	covar := [][]float64{{1}}
	pdf := gaussianPDF(x, mean, covar)
	if pdf != 0 {
		t.Errorf("expected 0 for dim mismatch, got %f", pdf)
	}
}

func TestGaussianPDF_FarFromMean(t *testing.T) {
	mean := []float64{0, 0}
	covar := identityMatrix(2)
	x := []float64{100, 100}

	pdf := gaussianPDF(x, mean, covar)
	if pdf > 1e-10 {
		t.Errorf("PDF far from mean should be near 0, got %f", pdf)
	}
}

func TestRegularizeCovariance(t *testing.T) {
	cov := [][]float64{{1, 0}, {0, 1}}
	regularizeCovariance(cov, 2, 0.1)
	if cov[0][0] != 1.1 || cov[1][1] != 1.1 {
		t.Errorf("regularization failed: %v", cov)
	}
	if cov[0][1] != 0 || cov[1][0] != 0 {
		t.Errorf("off-diagonal should not be modified: %v", cov)
	}
}

func TestKmeansPPInit(t *testing.T) {
	data := generateCluster([]float64{0, 0}, 20, 0.1)
	data = append(data, generateCluster([]float64{10, 10}, 20, 0.1)...)

	centers := kmeansPPInit(data, 2)
	if len(centers) != 2 {
		t.Fatalf("expected 2 centers, got %d", len(centers))
	}

	// Centers should be distinct
	dist := 0.0
	for i, c := range centers[0] {
		d := c - centers[1][i]
		dist += d * d
	}
	if dist < 1 {
		t.Errorf("centers too close: dist = %f", math.Sqrt(dist))
	}
}

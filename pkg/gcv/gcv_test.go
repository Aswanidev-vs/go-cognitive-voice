package gcv

import (
	"math"
	"testing"
)

func generateCluster(center []float64, n int, noise float64) [][]float64 {
	features := make([][]float64, n)
	for i := range features {
		features[i] = make([]float64, len(center))
		for j, c := range center {
			features[i][j] = c + math.Sin(float64(i*j+1))*noise
		}
	}
	return features
}

func TestVersion(t *testing.T) {
	if Version() != "0.1.0" {
		t.Errorf("Version: got %s, want 0.1.0", Version())
	}
}

func TestNewEngine(t *testing.T) {
	cfg := DefaultConfig()
	engine := NewEngine(cfg)
	if engine == nil {
		t.Fatal("engine should not be nil")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()
	if cfg.FeatureDim != 39 {
		t.Errorf("FeatureDim: got %d, want 39", cfg.FeatureDim)
	}
}

func TestEnrollAndIdentify(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TrainMethod = "centroid"
	engine := NewEngine(cfg)

	dim := cfg.FeatureDim
	aliceCenter := make([]float64, dim)
	bobCenter := make([]float64, dim)
	for i := 0; i < dim; i++ {
		if i%2 == 0 {
			aliceCenter[i] = 1.0
			bobCenter[i] = -1.0
		} else {
			aliceCenter[i] = 0.5
			bobCenter[i] = 2.0
		}
	}

	alice := generateCluster(aliceCenter, 30, 0.1)
	bob := generateCluster(bobCenter, 30, 0.1)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", alice)
	engine.EnrollSpeakerFromFeatures("bob", "Bob", bob)

	query := generateCluster(aliceCenter, 5, 0.05)

	result := engine.IdentifyFromFeatures(query)

	if !result.Identified {
		t.Error("should have identified a speaker")
	}
	if result.SpeakerID != "alice" {
		t.Errorf("identified: got %s, want alice", result.SpeakerID)
	}
}

func TestDetectFromFeatures(t *testing.T) {
	cfg := DefaultConfig()
	cfg.TrainMethod = "centroid"
	engine := NewEngine(cfg)

	dim := cfg.FeatureDim
	center := make([]float64, dim)
	for i := range center {
		if i%2 == 0 {
			center[i] = 1.0
		} else {
			center[i] = 0.5
		}
	}

	alice := generateCluster(center, 20, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", alice)

	query := generateCluster(center, 5, 0.05)

	matched, score, err := engine.DetectFromFeatures(query, "alice")
	if err != nil {
		t.Fatalf("DetectFromFeatures: %v", err)
	}

	if !matched {
		t.Errorf("should match alice (score=%f)", score)
	}
}

func TestListSpeakers(t *testing.T) {
	cfg := DefaultConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("a", "A", generateCluster([]float64{1}, 5, 0.1))
	engine.EnrollSpeakerFromFeatures("b", "B", generateCluster([]float64{2}, 5, 0.1))

	list := engine.ListSpeakers()
	if len(list) != 2 {
		t.Errorf("expected 2 speakers, got %d", len(list))
	}
}

func TestRemoveSpeaker(t *testing.T) {
	cfg := DefaultConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("a", "A", generateCluster([]float64{1}, 5, 0.1))
	if !engine.RemoveSpeaker("a") {
		t.Error("should remove existing speaker")
	}
	if engine.RemoveSpeaker("a") {
		t.Error("should return false for already removed")
	}
}

func TestString(t *testing.T) {
	cfg := DefaultConfig()
	engine := NewEngine(cfg)
	s := engine.String()
	if s == "" {
		t.Error("String() should not be empty")
	}
}

func TestExtractMFCC(t *testing.T) {
	samples := make([]float64, 16000)
	for i := range samples {
		samples[i] = math.Sin(2*math.Pi*440*float64(i)/16000) * 0.5
	}

	features := ExtractMFCC(samples, 16000)
	if len(features) == 0 {
		t.Fatal("expected non-empty features")
	}
}

func flattenToMono(features [][]float64) []float64 {
	if len(features) == 0 {
		return nil
	}
	dim := len(features[0])
	result := make([]float64, dim)
	for _, f := range features {
		for i := 0; i < dim && i < len(f); i++ {
			result[i] += f[i]
		}
	}
	n := float64(len(features))
	for i := range result {
		result[i] /= n
	}
	return result
}

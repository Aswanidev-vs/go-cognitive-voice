package speaker

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestNewEngine(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)
	if engine == nil {
		t.Fatal("engine should not be nil")
	}
	if engine.dim != 39 {
		t.Errorf("dim: got %d, want 39", engine.dim)
	}
}

func TestNewEngine_CustomDim(t *testing.T) {
	cfg := EngineConfig{FeatureDim: 20}
	engine := NewEngine(cfg)
	if engine.dim != 20 {
		t.Errorf("dim: got %d, want 20", engine.dim)
	}
}

func TestNewEngine_ZeroDim(t *testing.T) {
	cfg := EngineConfig{FeatureDim: 0}
	engine := NewEngine(cfg)
	if engine.dim != 39 {
		t.Errorf("dim: got %d, want 39 (default)", engine.dim)
	}
}

func TestEnrollAndIdentify_Centroid(t *testing.T) {
	cfg := DefaultEngineConfig()
	cfg.TrainMethod = "centroid"
	engine := NewEngine(cfg)

	// Use orthogonal feature centers so cosine similarity can distinguish them
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

	aliceFeatures := generateCluster(aliceCenter, 30, 0.1)
	bobFeatures := generateCluster(bobCenter, 30, 0.1)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", aliceFeatures)
	engine.EnrollSpeakerFromFeatures("bob", "Bob", bobFeatures)

	query := generateCluster(aliceCenter, 5, 0.05)
	result := engine.IdentifyFromFeatures(query)

	if !result.Identified {
		t.Error("should have identified a speaker")
	}
	if result.SpeakerID != "alice" {
		t.Errorf("identified: got %s, want alice", result.SpeakerID)
	}
}

func TestEnrollAndIdentify_DTW(t *testing.T) {
	cfg := DefaultEngineConfig()
	cfg.TrainMethod = "dtw"
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

	aliceFeatures := generateCluster(aliceCenter, 20, 0.1)
	bobFeatures := generateCluster(bobCenter, 20, 0.1)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", aliceFeatures)
	engine.EnrollSpeakerFromFeatures("bob", "Bob", bobFeatures)

	query := generateCluster(aliceCenter, 10, 0.05)
	result := engine.IdentifyFromFeatures(query)

	if !result.Identified {
		t.Error("should have identified a speaker")
	}
}

func TestIdentify_NoSpeakers(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	result := engine.IdentifyFromFeatures([][]float64{{1, 2, 3, 4, 5}})
	if result.Identified {
		t.Error("should not identify any speaker in empty database")
	}
}

func TestIdentify_EmptySamples(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2}, 10, 0.1))

	result := engine.IdentifyFromFeatures([][]float64{})
	if result.Identified {
		t.Error("should not identify with empty features")
	}
}

func TestDetect_Centroid(t *testing.T) {
	cfg := DefaultEngineConfig()
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

	aliceFeatures := generateCluster(center, 20, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", aliceFeatures)

	query := generateCluster(center, 5, 0.05)

	matched, score, err := engine.DetectFromFeatures(query, "alice")
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if !matched {
		t.Errorf("should match alice (score=%f)", score)
	}
}

func TestDetect_WrongSpeaker(t *testing.T) {
	cfg := DefaultEngineConfig()
	cfg.TrainMethod = "centroid"
	cfg.MatchThreshold = 0.99
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

	aliceFeatures := generateCluster(aliceCenter, 20, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", aliceFeatures)

	query := generateCluster(bobCenter, 5, 0.05)

	matched, score, err := engine.DetectFromFeatures(query, "alice")
	if err != nil {
		t.Fatalf("Detect: %v", err)
	}

	if matched {
		t.Errorf("should not match alice with score %f", score)
	}
}

func TestDetect_NonexistentSpeaker(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	_, _, err := engine.DetectFromFeatures([][]float64{{1, 2, 3}}, "nobody")
	if err == nil {
		t.Error("expected error for nonexistent speaker")
	}
}

func TestRemoveSpeaker(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2}, 10, 0.1))

	if !engine.RemoveSpeaker("alice") {
		t.Error("should return true for existing speaker")
	}
	if engine.RemoveSpeaker("alice") {
		t.Error("should return false for already removed speaker")
	}
	if engine.RemoveSpeaker("nobody") {
		t.Error("should return false for nonexistent speaker")
	}
}

func TestListSpeakers(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2}, 10, 0.1))
	engine.EnrollSpeakerFromFeatures("bob", "Bob", generateCluster([]float64{3, 4}, 10, 0.1))

	list := engine.ListSpeakers()
	if len(list) != 2 {
		t.Fatalf("expected 2 speakers, got %d", len(list))
	}

	ids := map[string]bool{}
	for _, s := range list {
		ids[s.ID] = true
	}
	if !ids["alice"] || !ids["bob"] {
		t.Errorf("expected alice and bob, got %v", ids)
	}
}

func TestSaveLoadDatabase(t *testing.T) {
	dir := t.TempDir()

	// Create and save
	cfg := DefaultEngineConfig()
	engine1 := NewEngine(cfg)
	engine1.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2, 3}, 20, 0.1))
	engine1.EnrollSpeakerFromFeatures("bob", "Bob", generateCluster([]float64{4, 5, 6}, 20, 0.1))

	if err := engine1.SaveDatabase(dir); err != nil {
		t.Fatalf("SaveDatabase: %v", err)
	}

	// Verify files were created
	entries, _ := os.ReadDir(dir)
	jsonCount := 0
	for _, e := range entries {
		if filepath.Ext(e.Name()) == ".json" {
			jsonCount++
		}
	}
	if jsonCount != 2 {
		t.Errorf("expected 2 JSON files, got %d", jsonCount)
	}

	// Load into new engine
	engine2 := NewEngine(cfg)
	if err := engine2.LoadDatabase(dir); err != nil {
		t.Fatalf("LoadDatabase: %v", err)
	}

	list := engine2.ListSpeakers()
	if len(list) != 2 {
		t.Fatalf("expected 2 speakers after load, got %d", len(list))
	}
}

func TestLoadDatabase_NonexistentDir(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	// Should not error on nonexistent dir
	err := engine.LoadDatabase("/nonexistent/dir")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestGetModel(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2}, 10, 0.1))

	model := engine.GetModel("alice")
	if model == nil {
		t.Error("GetModel returned nil")
	}
	if model.ID != "alice" {
		t.Errorf("ID: got %s, want alice", model.ID)
	}

	model = engine.GetModel("nobody")
	if model != nil {
		t.Error("GetModel should return nil for nonexistent speaker")
	}
}

func TestEnrollSpeaker_EmptyFeatures(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	err := engine.EnrollSpeaker("alice", "Alice", []string{"/nonexistent/file.wav"})
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestMultipleEnrollments(t *testing.T) {
	cfg := DefaultEngineConfig()
	cfg.TrainMethod = "centroid"
	engine := NewEngine(cfg)

	// Enroll same speaker twice (should update)
	features1 := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", features1)

	features2 := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice V2", features2)

	list := engine.ListSpeakers()
	if len(list) != 1 {
		t.Fatalf("expected 1 speaker, got %d", len(list))
	}
	if list[0].Name != "Alice V2" {
		t.Errorf("Name: got %s, want Alice V2", list[0].Name)
	}
}

func TestIdentifyFromFeatures_NaN(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2, 3}, 20, 0.1))

	// NaN features
	query := [][]float64{{math.NaN(), math.NaN(), math.NaN()}}
	result := engine.IdentifyFromFeatures(query)
	// Should not crash, may or may not identify
	_ = result
}

func TestScoredSpeaker(t *testing.T) {
	results := []ScoredSpeaker{
		{"c", 0.3},
		{"a", 0.9},
		{"b", 0.6},
	}
	sortScored(results)
	if results[0].SpeakerID != "a" || results[0].Score != 0.9 {
		t.Errorf("sort failed: %v", results)
	}
	if results[2].SpeakerID != "c" || results[2].Score != 0.3 {
		t.Errorf("sort failed: %v", results)
	}
}

func TestIdentify_TopN(t *testing.T) {
	cfg := DefaultEngineConfig()
	cfg.TrainMethod = "centroid"
	cfg.TopN = 2
	engine := NewEngine(cfg)

	var firstCenter []float64
	for i := 0; i < 4; i++ {
		center := make([]float64, cfg.FeatureDim)
		for j := range center {
			center[j] = float64((i+1)*(j%3+1)) + float64(i)*10
		}
		if i == 0 {
			firstCenter = center
		}
		id := fmt.Sprintf("spk%d", i)
		engine.EnrollSpeakerFromFeatures(id, id, generateCluster(center, 10, 0.01))
	}

	result := engine.IdentifyFromFeatures(generateCluster(firstCenter, 5, 0.01))
	if len(result.AllMatches) != cfg.TopN {
		t.Errorf("TopN=%d should cap AllMatches, got %d", cfg.TopN, len(result.AllMatches))
	}
	if !result.Identified || result.SpeakerID != "spk0" {
		t.Errorf("expected spk0, got %q (identified=%v)", result.SpeakerID, result.Identified)
	}
}

func TestEnrollAndIdentify_GMM(t *testing.T) {
	cfg := DefaultEngineConfig() // FeatureDim is 39 ...
	cfg.TrainMethod = "gmm"
	engine := NewEngine(cfg)

	// ... but real extraction produces (NumMFCC+1)*3 = 42 dims per frame.
	// The GMM must train on the actual dimension, not the configured one.
	featDim := (cfg.NumMFCC + 1) * 3
	aliceCenter := make([]float64, featDim)
	bobCenter := make([]float64, featDim)
	for i := 0; i < featDim; i++ {
		if i%2 == 0 {
			aliceCenter[i] = 1.0
			bobCenter[i] = -1.0
		} else {
			aliceCenter[i] = 0.5
			bobCenter[i] = 2.0
		}
	}

	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster(aliceCenter, 40, 0.1))
	engine.EnrollSpeakerFromFeatures("bob", "Bob", generateCluster(bobCenter, 40, 0.1))

	result := engine.IdentifyFromFeatures(generateCluster(aliceCenter, 5, 0.05))
	if len(result.AllMatches) == 0 {
		t.Fatal("expected at least one candidate")
	}
	best := result.AllMatches[0]
	if best.Score <= 0 {
		t.Fatalf("GMM model did not learn anything (score=%f)", best.Score)
	}
	if !result.Identified {
		t.Errorf("should have identified alice (score=%f, threshold=%f)", best.Score, cfg.MatchThreshold)
	}
	if best.SpeakerID != "alice" {
		t.Errorf("identified: got %s, want alice", best.SpeakerID)
	}
}

func TestSaveDatabase_RemovesStaleFiles(t *testing.T) {
	dir := t.TempDir()
	cfg := DefaultEngineConfig()

	engine := NewEngine(cfg)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", generateCluster([]float64{1, 2, 3}, 10, 0.1))
	engine.EnrollSpeakerFromFeatures("bob", "Bob", generateCluster([]float64{4, 5, 6}, 10, 0.1))
	if err := engine.SaveDatabase(dir); err != nil {
		t.Fatalf("SaveDatabase: %v", err)
	}

	engine.RemoveSpeaker("bob")
	if err := engine.SaveDatabase(dir); err != nil {
		t.Fatalf("SaveDatabase after remove: %v", err)
	}

	loaded := NewEngine(cfg)
	if err := loaded.LoadDatabase(dir); err != nil {
		t.Fatalf("LoadDatabase: %v", err)
	}
	list := loaded.ListSpeakers()
	if len(list) != 1 || list[0].ID != "alice" {
		t.Errorf("removed speaker resurrected: got %v, want only alice", list)
	}
}

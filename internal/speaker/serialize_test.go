package speaker

import (
	"os"
	"path/filepath"
	"testing"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.json")

	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 20, 0.1)
	m.TrainCentroid(features)

	if err := saveModel(m, path); err != nil {
		t.Fatalf("saveModel: %v", err)
	}

	loaded, err := loadModel(path)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	if loaded.ID != m.ID {
		t.Errorf("ID: got %s, want %s", loaded.ID, m.ID)
	}
	if loaded.Name != m.Name {
		t.Errorf("Name: got %s, want %s", loaded.Name, m.Name)
	}
	if loaded.Dim != m.Dim {
		t.Errorf("Dim: got %d, want %d", loaded.Dim, m.Dim)
	}
	if loaded.TrainMethod != m.TrainMethod {
		t.Errorf("TrainMethod: got %s, want %s", loaded.TrainMethod, m.TrainMethod)
	}
	if loaded.NumSamples != m.NumSamples {
		t.Errorf("NumSamples: got %d, want %d", loaded.NumSamples, m.NumSamples)
	}

	// Check centroid
	if len(loaded.Centroid) != len(m.Centroid) {
		t.Fatalf("Centroid length: got %d, want %d", len(loaded.Centroid), len(m.Centroid))
	}
	for i := range loaded.Centroid {
		if loaded.Centroid[i] != m.Centroid[i] {
			t.Errorf("Centroid[%d]: got %f, want %f", i, loaded.Centroid[i], m.Centroid[i])
		}
	}

	// Check features
	if len(loaded.Features) != len(m.Features) {
		t.Errorf("Features length: got %d, want %d", len(loaded.Features), len(m.Features))
	}
}

func TestSaveLoadGMM(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_gmm.json")

	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 50, 0.2)
	m.TrainGMM(features, 3, 50, 1e-6)

	if err := saveModel(m, path); err != nil {
		t.Fatalf("saveModel: %v", err)
	}

	loaded, err := loadModel(path)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	if loaded.GMM == nil {
		t.Fatal("GMM should not be nil after load")
	}
	if loaded.GMM.K != m.GMM.K {
		t.Errorf("GMM.K: got %d, want %d", loaded.GMM.K, m.GMM.K)
	}
	if loaded.GMM.Dim != m.GMM.Dim {
		t.Errorf("GMM.Dim: got %d, want %d", loaded.GMM.Dim, m.GMM.Dim)
	}
}

func TestSaveLoadDTW(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test_dtw.json")

	m := NewSpeakerModel("spk1", "Alice", 3)
	features := generateCluster([]float64{1, 2, 3}, 10, 0.1)
	m.TrainDTW(features)

	if err := saveModel(m, path); err != nil {
		t.Fatalf("saveModel: %v", err)
	}

	loaded, err := loadModel(path)
	if err != nil {
		t.Fatalf("loadModel: %v", err)
	}

	if loaded.TrainMethod != "dtw" {
		t.Errorf("TrainMethod: got %s, want dtw", loaded.TrainMethod)
	}
	if len(loaded.Features) != len(features) {
		t.Errorf("Features length: got %d, want %d", len(loaded.Features), len(features))
	}
}

func TestLoadModel_InvalidJSON(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	os.WriteFile(path, []byte("not json"), 0644)

	_, err := loadModel(path)
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoadModel_Nonexistent(t *testing.T) {
	_, err := loadModel("/nonexistent/path.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestSaveModel_InvalidDir(t *testing.T) {
	m := NewSpeakerModel("spk1", "Alice", 3)
	err := saveModel(m, "/nonexistent/dir/test.json")
	if err == nil {
		t.Error("expected error for invalid directory")
	}
}

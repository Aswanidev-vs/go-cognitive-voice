package neural

import (
	"math"
	"testing"
)

func TestNew(t *testing.T) {
	n := New(42, []int{128, 64}, 32, ActLeakyReLU)
	if n == nil {
		t.Fatal("network should not be nil")
	}
	if n.dimIn != 42 {
		t.Errorf("dimIn: got %d, want 42", n.dimIn)
	}
	if n.dimEmbed != 32 {
		t.Errorf("dimEmbed: got %d, want 32", n.dimEmbed)
	}
	if len(n.layers) != 3 {
		t.Errorf("layers: got %d, want 3", len(n.layers))
	}
}

func TestForward(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)
	input := make([]float64, 42)
	for i := range input {
		input[i] = float64(i) * 0.1
	}

	output := n.Forward(input)
	if len(output) != 32 {
		t.Fatalf("output dim: got %d, want 32", len(output))
	}

	// All outputs should be finite
	for i, v := range output {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("output[%d] = %f, not finite", i, v)
		}
	}
}

func TestForwardEmbed_Normalized(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)
	input := make([]float64, 42)
	for i := range input {
		input[i] = float64(i)
	}

	emb := n.ForwardEmbed(input)
	norm := 0.0
	for _, v := range emb {
		norm += v * v
	}
	norm = math.Sqrt(norm)

	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("embedding norm: got %f, want ~1.0", norm)
	}
}

func TestEmbed_Average(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)
	features := make([][]float64, 10)
	for i := range features {
		features[i] = make([]float64, 42)
		for j := range features[i] {
			features[i][j] = float64(i+j) * 0.01
		}
	}

	emb := n.Embed(features)
	if len(emb) != 32 {
		t.Fatalf("embedding dim: got %d, want 32", len(emb))
	}

	// Should be normalized
	norm := 0.0
	for _, v := range emb {
		norm += v * v
	}
	if math.Abs(math.Sqrt(norm)-1.0) > 1e-6 {
		t.Error("embedding should be normalized")
	}
}

func TestEmbed_EmptyFeatures(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)
	emb := n.Embed(nil)
	if len(emb) != 32 {
		t.Errorf("expected 32, got %d", len(emb))
	}
}

func TestTrain_SingleSpeaker(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)

	// Single speaker — can't do contrastive learning
	speakerFeats := map[string][][]float64{
		"alice": make([][]float64, 20),
	}
	for i := range speakerFeats["alice"] {
		speakerFeats["alice"][i] = make([]float64, 42)
		for j := range speakerFeats["alice"][i] {
			speakerFeats["alice"][i][j] = math.Sin(float64(i*j)) * 0.5
		}
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 5
	loss := n.Train(speakerFeats, cfg)

	if loss != 0 {
		t.Errorf("single speaker should return 0 loss, got %f", loss)
	}
}

func TestTrain_MultipleSpeakers(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)

	speakerFeats := map[string][][]float64{
		"alice": make([][]float64, 15),
		"bob":   make([][]float64, 15),
	}

	for i := range speakerFeats["alice"] {
		speakerFeats["alice"][i] = make([]float64, 42)
		for j := range speakerFeats["alice"][i] {
			speakerFeats["alice"][i][j] = math.Sin(float64(i+j)) * 0.5
		}
	}
	for i := range speakerFeats["bob"] {
		speakerFeats["bob"][i] = make([]float64, 42)
		for j := range speakerFeats["bob"][i] {
			speakerFeats["bob"][i][j] = math.Cos(float64(i+j)) * 0.5
		}
	}

	cfg := DefaultTrainConfig()
	cfg.Epochs = 10
	loss := n.Train(speakerFeats, cfg)

	if loss < 0 {
		t.Errorf("loss should be non-negative, got %f", loss)
	}
}

func TestPredict(t *testing.T) {
	n := New(42, []int{64}, 32, ActLeakyReLU)

	enrolled := map[string][]float64{
		"alice": make([]float64, 32),
		"bob":   make([]float64, 32),
	}
	for i := range enrolled["alice"] {
		enrolled["alice"][i] = math.Sin(float64(i)) * 0.5
		enrolled["bob"][i] = math.Cos(float64(i)) * 0.5
	}

	features := make([][]float64, 5)
	for i := range features {
		features[i] = make([]float64, 42)
		for j := range features[i] {
			features[i][j] = math.Sin(float64(i+j)) * 0.5
		}
	}

	id, score := n.Predict(features, enrolled)
	if id == "" {
		t.Error("predicted ID should not be empty")
	}
	if score < 0 || score > 1 {
		t.Errorf("score should be in [0,1], got %f", score)
	}
}

func TestDefaultTrainConfig(t *testing.T) {
	cfg := DefaultTrainConfig()
	if cfg.Epochs != 100 {
		t.Errorf("Epochs: got %d, want 100", cfg.Epochs)
	}
	if cfg.LR != 0.001 {
		t.Errorf("LR: got %f, want 0.001", cfg.LR)
	}
}

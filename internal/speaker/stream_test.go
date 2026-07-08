package speaker

import (
	"testing"
)

func TestStreamBuffer_Basic(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	alice := generateCluster([]float64{1, 2, 3, 4, 5}, 30, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", alice)

	// chunk = 1 second at 16kHz = 16000 samples
	buf := NewStreamBuffer(engine, 16000, 1.0)

	// Write less than one chunk
	chunk := make([]float64, 8000)
	for i := range chunk {
		chunk[i] = 0.5
	}
	buf.Write(chunk)

	_, ok := buf.Identify()
	if ok {
		t.Error("should not identify with less than one chunk")
	}

	// Write another chunk to exceed threshold
	buf.Write(chunk)

	result, ok := buf.Identify()
	if !ok {
		t.Error("should identify after accumulating enough audio")
	}
	if result == nil {
		t.Fatal("result should not be nil")
	}
}

func TestStreamBuffer_IdentifyFull(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	alice := generateCluster([]float64{1, 2, 3, 4, 5}, 30, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", alice)

	buf := NewStreamBuffer(engine, 16000, 10.0) // large chunk

	// Write small amount
	samples := make([]float64, 8000)
	for i := range samples {
		samples[i] = 0.5
	}
	buf.Write(samples)

	// IdentifyFull should still work on partial audio
	result := buf.IdentifyFull()
	if result == nil {
		t.Fatal("IdentifyFull should return a result")
	}
}

func TestStreamBuffer_Reset(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)
	buf := NewStreamBuffer(engine, 16000, 1.0)

	samples := make([]float64, 16000)
	buf.Write(samples)

	if buf.Samples() != 16000 {
		t.Errorf("expected 16000 samples, got %d", buf.Samples())
	}

	buf.Reset()

	if buf.Samples() != 0 {
		t.Errorf("expected 0 after reset, got %d", buf.Samples())
	}
}

func TestStreamBuffer_Duration(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)
	buf := NewStreamBuffer(engine, 16000, 1.0)

	samples := make([]float64, 32000) // 2 seconds
	buf.Write(samples)

	if buf.Duration() != 2.0 {
		t.Errorf("expected 2.0s duration, got %f", buf.Duration())
	}
}

func TestStreamBuffer_MultipleChunks(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)

	alice := generateCluster([]float64{1, 2, 3, 4, 5}, 30, 0.1)
	engine.EnrollSpeakerFromFeatures("alice", "Alice", alice)

	buf := NewStreamBuffer(engine, 16000, 0.5) // 0.5s chunks

	identified := 0
	for i := 0; i < 10; i++ {
		chunk := make([]float64, 8000) // 0.5s
		for j := range chunk {
			chunk[j] = 0.5 + float64(j)*0.0001
		}
		buf.Write(chunk)

		if _, ok := buf.Identify(); ok {
			identified++
		}
	}

	if identified == 0 {
		t.Error("should have identified at least once in 10 chunks")
	}
}

func TestStreamBuffer_DefaultChunk(t *testing.T) {
	cfg := DefaultEngineConfig()
	engine := NewEngine(cfg)
	buf := NewStreamBuffer(engine, 16000, 0) // should default to 2.0

	if buf.chunkSec != 2.0 {
		t.Errorf("expected default chunkSec=2.0, got %f", buf.chunkSec)
	}
}

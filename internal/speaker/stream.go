package speaker

// StreamBuffer accumulates audio samples from a live stream and identifies
// speakers in chunks. It is NOT safe for concurrent use — the caller must
// serialize calls to Write and Identify.
type StreamBuffer struct {
	engine     *Engine
	samples    []float64
	sampleRate int
	chunkSec   float64 // seconds per identification chunk
}

// NewStreamBuffer creates a streaming buffer that identifies speakers
// every chunkSec seconds of accumulated audio.
//
//	chunkSec = 2.0 means identify every ~2 seconds of audio
//	chunkSec = 0.5 means identify every ~0.5 seconds (more responsive, more CPU)
func NewStreamBuffer(engine *Engine, sampleRate int, chunkSec float64) *StreamBuffer {
	if chunkSec <= 0 {
		chunkSec = 2.0
	}
	return &StreamBuffer{
		engine:     engine,
		sampleRate: sampleRate,
		chunkSec:   chunkSec,
	}
}

// Write pushes audio samples into the buffer. Samples should be mono
// float64 values normalized to [-1, 1]. Push from any source: microphone,
// websocket, file chunker, etc.
func (s *StreamBuffer) Write(samples []float64) {
	s.samples = append(s.samples, samples...)
}

// Identify checks if enough audio has accumulated and runs identification.
// Returns (result, true) when a chunk was processed, or (nil, false) if
// not enough audio yet.
//
// Typical loop:
//
//	for chunk := range audioChan {
//	    buf.Write(chunk)
//	    if result, ok := buf.Identify(); ok {
//	        fmt.Println(result.SpeakerName, result.Score)
//	    }
//	}
func (s *StreamBuffer) Identify() (*MatchResult, bool) {
	chunkSamples := int(s.chunkSec * float64(s.sampleRate))

	if len(s.samples) < chunkSamples {
		return nil, false
	}

	// Take the most recent chunk (sliding window)
	start := len(s.samples) - chunkSamples
	chunk := s.samples[start:]

	s.engine.mu.RLock()
	defer s.engine.mu.RUnlock()

	// Extract features
	features := s.engine.extractFeaturesFromSamples(chunk, s.sampleRate)
	if len(features) == 0 {
		return nil, false
	}

	result := s.engine.identifyFromFeatures(features)
	return result, true
}

// IdentifyFull runs identification on ALL accumulated samples (not just
// the latest chunk). Use this when the stream ends and you want a final
// identification on the full buffer.
func (s *StreamBuffer) IdentifyFull() *MatchResult {
	if len(s.samples) == 0 {
		return &MatchResult{Identified: false}
	}

	s.engine.mu.RLock()
	defer s.engine.mu.RUnlock()

	features := s.engine.extractFeaturesFromSamples(s.samples, s.sampleRate)
	if len(features) == 0 {
		return &MatchResult{Identified: false}
	}

	return s.engine.identifyFromFeatures(features)
}

// Samples returns the number of samples currently buffered.
func (s *StreamBuffer) Samples() int {
	return len(s.samples)
}

// Duration returns the duration of buffered audio in seconds.
func (s *StreamBuffer) Duration() float64 {
	return float64(len(s.samples)) / float64(s.sampleRate)
}

// Reset clears the buffer.
func (s *StreamBuffer) Reset() {
	s.samples = s.samples[:0]
}

// Package gcv provides speaker detection and identification in pure Go.
//
// It supports WAV and MP3 audio formats, extracts MFCC features, and uses
// multiple matching strategies (centroid, GMM, DTW) for speaker identification.
//
// Basic usage:
//
//	engine := gcv.NewEngine(gcv.DefaultConfig())
//	engine.EnrollSpeaker("spk1", "Alice", []string{"alice1.wav", "alice2.wav"})
//	result, _ := engine.Identify("unknown.wav")
//	if result.Identified {
//	    fmt.Printf("Speaker: %s (score: %.3f)\n", result.SpeakerName, result.Score)
//	}
package gcv

import (
	"fmt"

	"github.com/Aswanidev-vs/go-cognitive-voice/internal/audio"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/dsp"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/speaker"
)

// Config holds the engine configuration.
type Config = speaker.EngineConfig

// DefaultConfig returns default configuration.
func DefaultConfig() Config {
	return speaker.DefaultEngineConfig()
}

// Result holds the identification result.
type Result = speaker.MatchResult

// SpeakerInfo holds enrolled speaker info.
type SpeakerInfo = speaker.SpeakerInfo

// Engine is the main API for speaker detection and identification.
type Engine struct {
	inner *speaker.Engine
}

// NewEngine creates a new speaker engine.
func NewEngine(cfg Config) *Engine {
	return &Engine{inner: speaker.NewEngine(cfg)}
}

// EnrollSpeaker enrolls a speaker from audio files.
// Provide one or more audio files (WAV/MP3) for better accuracy.
func (e *Engine) EnrollSpeaker(id, name string, audioPaths []string) error {
	return e.inner.EnrollSpeaker(id, name, audioPaths)
}

// EnrollSpeakerFromFeatures enrolls a speaker from pre-extracted features.
func (e *Engine) EnrollSpeakerFromFeatures(id, name string, features [][]float64) {
	e.inner.EnrollSpeakerFromFeatures(id, name, features)
}

// Identify identifies the speaker in an audio file.
// Returns the best match and all scored candidates.
func (e *Engine) Identify(audioPath string) (*Result, error) {
	return e.inner.Identify(audioPath)
}

// IdentifyFromSamples identifies the speaker from raw audio samples.
func (e *Engine) IdentifyFromSamples(samples []float64, sampleRate int) (*Result, error) {
	return e.inner.IdentifyFromSamples(samples, sampleRate)
}

// IdentifyFromFeatures identifies the speaker from pre-extracted features.
func (e *Engine) IdentifyFromFeatures(features [][]float64) *Result {
	return e.inner.IdentifyFromFeatures(features)
}

// DetectFromFeatures checks if pre-extracted features match a specific speaker.
func (e *Engine) DetectFromFeatures(features [][]float64, speakerID string) (bool, float64, error) {
	return e.inner.DetectFromFeatures(features, speakerID)
}

// Detect checks if the audio matches a specific speaker.
// Returns (matched, score, error).
func (e *Engine) Detect(audioPath, speakerID string) (bool, float64, error) {
	return e.inner.Detect(audioPath, speakerID)
}

// DetectFromSamples checks if raw audio samples match a specific speaker.
func (e *Engine) DetectFromSamples(samples []float64, sampleRate int, speakerID string) (bool, float64, error) {
	return e.inner.DetectFromSamples(samples, sampleRate, speakerID)
}

// RemoveSpeaker removes a speaker from the database.
func (e *Engine) RemoveSpeaker(id string) bool {
	return e.inner.RemoveSpeaker(id)
}

// ListSpeakers returns all enrolled speakers.
func (e *Engine) ListSpeakers() []SpeakerInfo {
	return e.inner.ListSpeakers()
}

// SaveDatabase saves all enrolled speakers to a directory.
func (e *Engine) SaveDatabase(dir string) error {
	return e.inner.SaveDatabase(dir)
}

// LoadDatabase loads enrolled speakers from a directory.
func (e *Engine) LoadDatabase(dir string) error {
	return e.inner.LoadDatabase(dir)
}

// ExtractFeatures extracts MFCC features from an audio file.
func (e *Engine) ExtractFeatures(audioPath string) ([][]float64, error) {
	return e.inner.ExtractFeatures(audioPath)
}

// --- Streaming ---

// StreamBuffer accumulates audio from a live source and identifies speakers
// in fixed-size chunks. Not safe for concurrent use.
type StreamBuffer struct {
	inner *speaker.StreamBuffer
}

// NewStreamBuffer creates a streaming buffer.
//
//	sampleRate: audio sample rate (e.g. 16000)
//	chunkSec: seconds per identification chunk (e.g. 2.0)
func (e *Engine) NewStreamBuffer(sampleRate int, chunkSec float64) *StreamBuffer {
	return &StreamBuffer{inner: speaker.NewStreamBuffer(e.inner, sampleRate, chunkSec)}
}

// Write pushes audio samples. Values should be mono float64 in [-1, 1].
func (s *StreamBuffer) Write(samples []float64) { s.inner.Write(samples) }

// Identify checks if enough audio has accumulated and runs identification.
// Returns (result, true) when a chunk was processed, (nil, false) otherwise.
func (s *StreamBuffer) Identify() (*Result, bool) {
	r, ok := s.inner.Identify()
	return r, ok
}

// IdentifyFull identifies using ALL accumulated audio (use at stream end).
func (s *StreamBuffer) IdentifyFull() *Result { return s.inner.IdentifyFull() }

// Samples returns buffered sample count.
func (s *StreamBuffer) Samples() int { return s.inner.Samples() }

// Duration returns buffered audio duration in seconds.
func (s *StreamBuffer) Duration() float64 { return s.inner.Duration() }

// Reset clears the buffer.
func (s *StreamBuffer) Reset() { s.inner.Reset() }

// --- Convenience functions ---

// ReadAudio reads an audio file and returns normalized mono samples.
func ReadAudio(path string) ([]float64, int, error) {
	data, err := audio.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	return data.Samples, data.SampleRate, nil
}

// ExtractMFCC extracts MFCC features from raw audio samples.
func ExtractMFCC(samples []float64, sampleRate int) [][]float64 {
	result := dsp.ExtractMFCCFromRaw(samples, sampleRate)
	return dsp.ConcatenateFeatures(result)
}

// Version returns the library version.
func Version() string {
	return "0.1.0"
}

// String returns a description of the engine.
func (e *Engine) String() string {
	speakers := e.inner.ListSpeakers()
	return fmt.Sprintf("gcv engine v%s (%d speakers enrolled)", Version(), len(speakers))
}

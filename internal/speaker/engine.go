package speaker

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/Aswanidev-vs/go-cognitive-voice/internal/audio"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/dsp"
)

// Engine is the main speaker detection and identification engine.
type Engine struct {
	mu       sync.RWMutex
	speakers map[string]*SpeakerModel
	dim      int
	cfg      EngineConfig
}

// EngineConfig holds engine configuration.
// EngineConfig controls the speaker identification pipeline.
//
// Defaults work well for most cases. Only change what you need:
//
//	cfg := gcv.DefaultConfig()
//	cfg.MatchThreshold = 0.85  // stricter matching
//	engine := gcv.NewEngine(cfg)
type EngineConfig struct {
	// FeatureDim is the total feature vector size per frame.
	// Default 39 = 13 MFCC + 13 delta + 13 delta-delta.
	// Only change this if you know what you're doing.
	FeatureDim int

	// NumMFCC is the number of MFCC coefficients per frame.
	// Typical range: 12-13. Lower = faster, less detailed.
	NumMFCC int

	// NumFilters is the number of mel-spaced triangular filters.
	// Typical range: 20-40. More filters = finer frequency resolution.
	NumFilters int

	// FFTSize is the FFT window size (must be power of 2).
	// 512 works for 16kHz audio. Use 256 for 8kHz, 1024 for 32kHz+.
	FFTSize int

	// MatchThreshold is the minimum score to accept a match (0.0 to 1.0).
	// Lower = more false positives, higher = more false negatives.
	//   0.5  = lenient  (default, good for clean audio)
	//   0.7  = moderate (recommended for real-world audio)
	//   0.85 = strict   (fewer false positives)
	//   0.95 = very strict (almost no false positives)
	MatchThreshold float64

	// TopN is the number of top candidates returned in AllMatches.
	// Default 5. Set to 0 to return all.
	TopN int

	// TrainMethod selects the speaker model algorithm:
	//   "centroid" — fast, good for clean audio (default)
	//   "gmm"      — slower, better for noisy audio
	//   "dtw"      — best for variable-length sequences
	TrainMethod string

	// GMMComponents is the number of Gaussian mixture components.
	// Only used when TrainMethod is "gmm". Default 8.
	// More components = more expressive but slower.
	GMMComponents int
}

// DefaultEngineConfig returns default configuration.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		FeatureDim:     39,
		NumMFCC:        13,
		NumFilters:     26,
		FFTSize:        512,
		MatchThreshold: 0.5,
		TopN:           5,
		TrainMethod:    "centroid",
		GMMComponents:  8,
	}
}

// MatchResult holds the result of speaker identification.
type MatchResult struct {
	Identified   bool
	SpeakerID    string
	SpeakerName  string
	Score        float64
	AllMatches   []SpeakerMatch
}

// SpeakerMatch holds a single speaker match.
type SpeakerMatch struct {
	SpeakerID   string
	SpeakerName string
	Score       float64
}

// NewEngine creates a new speaker engine.
func NewEngine(cfg EngineConfig) *Engine {
	if cfg.FeatureDim <= 0 {
		cfg.FeatureDim = 39
	}
	return &Engine{
		speakers: make(map[string]*SpeakerModel),
		dim:      cfg.FeatureDim,
		cfg:      cfg,
	}
}

// EnrollSpeaker enrolls a speaker from one or more audio files.
func (e *Engine) EnrollSpeaker(id, name string, audioPaths []string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	var allFeatures [][]float64

	for _, path := range audioPaths {
		features, err := e.extractFeatures(path)
		if err != nil {
			return fmt.Errorf("extract features from %s: %w", path, err)
		}
		allFeatures = append(allFeatures, features...)
	}

	if len(allFeatures) == 0 {
		return fmt.Errorf("no features extracted from audio files")
	}

	model := NewSpeakerModel(id, name, e.dim)

	switch e.cfg.TrainMethod {
	case "gmm":
		model.TrainGMM(allFeatures, e.cfg.GMMComponents, 100, 1e-6)
	case "dtw":
		model.TrainDTW(allFeatures)
	default:
		model.TrainCentroid(allFeatures)
	}

	e.speakers[id] = model
	return nil
}

// EnrollSpeakerFromFeatures enrolls a speaker from pre-extracted features.
func (e *Engine) EnrollSpeakerFromFeatures(id, name string, features [][]float64) {
	e.mu.Lock()
	defer e.mu.Unlock()

	model := NewSpeakerModel(id, name, e.dim)

	switch e.cfg.TrainMethod {
	case "gmm":
		model.TrainGMM(features, e.cfg.GMMComponents, 100, 1e-6)
	case "dtw":
		model.TrainDTW(features)
	default:
		model.TrainCentroid(features)
	}

	e.speakers[id] = model
}

// Identify identifies the speaker in an audio file.
func (e *Engine) Identify(audioPath string) (*MatchResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	features, err := e.extractFeatures(audioPath)
	if err != nil {
		return nil, fmt.Errorf("extract features: %w", err)
	}

	if len(features) == 0 {
		return &MatchResult{Identified: false}, nil
	}

	return e.identifyFromFeatures(features), nil
}

// IdentifyFromSamples identifies the speaker from raw audio samples.
func (e *Engine) IdentifyFromSamples(samples []float64, sampleRate int) (*MatchResult, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	features := e.extractFeaturesFromSamples(samples, sampleRate)

	if len(features) == 0 {
		return &MatchResult{Identified: false}, nil
	}

	return e.identifyFromFeatures(features), nil
}

// IdentifyFromFeatures identifies the speaker from pre-extracted features.
func (e *Engine) IdentifyFromFeatures(features [][]float64) *MatchResult {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(features) == 0 {
		return &MatchResult{Identified: false}
	}

	return e.identifyFromFeatures(features)
}

func (e *Engine) identifyFromFeatures(features [][]float64) *MatchResult {
	if len(e.speakers) == 0 {
		return &MatchResult{Identified: false}
	}

	matches := make([]SpeakerMatch, 0, len(e.speakers))

	for _, model := range e.speakers {
		var score float64
		if e.cfg.TrainMethod == "dtw" {
			score = model.ScoreSequence(features)
		} else {
			// For centroid/gmm, compare against mean of query features
			meanFeatures := meanVector(features)
			score = model.Score(meanFeatures)
		}

		matches = append(matches, SpeakerMatch{
			SpeakerID:   model.ID,
			SpeakerName: model.Name,
			Score:       score,
		})
	}

	// Sort by score (highest first)
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Score > matches[j].Score
	})

	result := &MatchResult{
		AllMatches: matches,
	}

	if len(matches) > 0 && matches[0].Score >= e.cfg.MatchThreshold {
		result.Identified = true
		result.SpeakerID = matches[0].SpeakerID
		result.SpeakerName = matches[0].SpeakerName
		result.Score = matches[0].Score
	}

	return result
}

// Detect checks if the audio matches a specific speaker.
func (e *Engine) Detect(audioPath, speakerID string) (bool, float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	features, err := e.extractFeatures(audioPath)
	if err != nil {
		return false, 0, fmt.Errorf("extract features: %w", err)
	}

	if len(features) == 0 {
		return false, 0, nil
	}

	model, ok := e.speakers[speakerID]
	if !ok {
		return false, 0, fmt.Errorf("speaker %s not found", speakerID)
	}

	var score float64
	if e.cfg.TrainMethod == "dtw" {
		score = model.ScoreSequence(features)
	} else {
		meanFeatures := meanVector(features)
		score = model.Score(meanFeatures)
	}

	return score >= e.cfg.MatchThreshold, score, nil
}

// DetectFromFeatures checks if pre-extracted features match a specific speaker.
func (e *Engine) DetectFromFeatures(features [][]float64, speakerID string) (bool, float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if len(features) == 0 {
		return false, 0, nil
	}

	model, ok := e.speakers[speakerID]
	if !ok {
		return false, 0, fmt.Errorf("speaker %s not found", speakerID)
	}

	var score float64
	if e.cfg.TrainMethod == "dtw" {
		score = model.ScoreSequence(features)
	} else {
		meanFeatures := meanVector(features)
		score = model.Score(meanFeatures)
	}

	return score >= e.cfg.MatchThreshold, score, nil
}

// DetectFromSamples checks if raw audio samples match a specific speaker.
func (e *Engine) DetectFromSamples(samples []float64, sampleRate int, speakerID string) (bool, float64, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	features := e.extractFeaturesFromSamples(samples, sampleRate)
	if len(features) == 0 {
		return false, 0, nil
	}

	model, ok := e.speakers[speakerID]
	if !ok {
		return false, 0, fmt.Errorf("speaker %s not found", speakerID)
	}

	var score float64
	if e.cfg.TrainMethod == "dtw" {
		score = model.ScoreSequence(features)
	} else {
		meanFeatures := meanVector(features)
		score = model.Score(meanFeatures)
	}

	return score >= e.cfg.MatchThreshold, score, nil
}

// RemoveSpeaker removes a speaker from the database.
func (e *Engine) RemoveSpeaker(id string) bool {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, ok := e.speakers[id]; ok {
		delete(e.speakers, id)
		return true
	}
	return false
}

// ListSpeakers returns all enrolled speaker IDs and names.
func (e *Engine) ListSpeakers() []SpeakerInfo {
	e.mu.RLock()
	defer e.mu.RUnlock()

	list := make([]SpeakerInfo, 0, len(e.speakers))
	for _, model := range e.speakers {
		list = append(list, SpeakerInfo{
			ID:         model.ID,
			Name:       model.Name,
			NumSamples: model.NumSamples,
			TrainMethod: model.TrainMethod,
		})
	}
	return list
}

// SpeakerInfo holds basic info about an enrolled speaker.
type SpeakerInfo struct {
	ID          string
	Name        string
	NumSamples  int
	TrainMethod string
}

// ExtractFeatures extracts MFCC features from an audio file.
func (e *Engine) ExtractFeatures(audioPath string) ([][]float64, error) {
	return e.extractFeatures(audioPath)
}

func (e *Engine) extractFeatures(path string) ([][]float64, error) {
	audioData, err := audio.ReadFile(path)
	if err != nil {
		return nil, err
	}

	return e.extractFeaturesFromSamples(audioData.Samples, audioData.SampleRate), nil
}

func (e *Engine) extractFeaturesFromSamples(samples []float64, sampleRate int) [][]float64 {
	// Extract MFCC with deltas
	result := dsp.ExtractMFCCFromRaw(samples, sampleRate)

	// Concatenate MFCC + deltas + delta-deltas
	features := dsp.ConcatenateFeatures(result)

	return features
}

// SaveDatabase saves enrolled speakers to a directory.
func (e *Engine) SaveDatabase(dir string) error {
	e.mu.RLock()
	defer e.mu.RUnlock()

	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("create db dir: %w", err)
	}

	for _, model := range e.speakers {
		path := filepath.Join(dir, model.ID+".json")
		if err := saveModel(model, path); err != nil {
			return fmt.Errorf("save speaker %s: %w", model.ID, err)
		}
	}

	return nil
}

// LoadDatabase loads enrolled speakers from a directory.
func (e *Engine) LoadDatabase(dir string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("read db dir: %w", err)
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}

		path := filepath.Join(dir, entry.Name())
		model, err := loadModel(path)
		if err != nil {
			return fmt.Errorf("load speaker %s: %w", entry.Name(), err)
		}

		e.speakers[model.ID] = model
	}

	return nil
}

// GetModel returns a speaker model (for advanced usage).
func (e *Engine) GetModel(id string) *SpeakerModel {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.speakers[id]
}

func meanVector(features [][]float64) []float64 {
	if len(features) == 0 {
		return nil
	}
	dim := len(features[0])
	mean := make([]float64, dim)
	for _, f := range features {
		for i := 0; i < dim && i < len(f); i++ {
			mean[i] += f[i]
		}
	}
	n := float64(len(features))
	for i := range mean {
		mean[i] /= n
	}
	return mean
}

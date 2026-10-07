package speaker

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/Aswanidev-vs/go-cognitive-voice/internal/audio"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/dsp"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/ivector"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/matcher"
	"github.com/Aswanidev-vs/go-cognitive-voice/internal/neural"
)

// Engine is the main speaker detection and identification engine.
type Engine struct {
	mu       sync.RWMutex
	speakers map[string]*SpeakerModel
	dim      int
	cfg      EngineConfig

	// Neural network for learned embeddings
	nn         *neural.Network
	nnEnrolled map[string][]float64 // speaker ID → embedding

	// i-vector + PLDA
	ivModel    *ivector.IVectorModel
	plda       *ivector.PLDAModel
	ivEnrolled map[string][]float64 // speaker ID → i-vector
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
	// FeatureDim is the fallback feature vector size used before any
	// features have been observed (model metadata, neural input guess).
	// Actual extraction produces (NumMFCC+1)×3 values per frame
	// (energy + MFCC, delta, delta-delta) = 42 by default.
	// Only change this if you know what you're doing.
	FeatureDim int

	// NumMFCC is the number of MFCC coefficients per frame.
	// Typical range: 12-13. Lower = faster, less detailed.
	NumMFCC int

	// NumFilters is the number of mel-spaced triangular filters.
	// Typical range: 20-40. More filters = finer frequency resolution.
	NumFilters int

	// FFTSize is the FFT window size (must be power of 2).
	// 0 = derive from the sample rate via the 25 ms frame (256 @ 8 kHz,
	// 512 @ 16 kHz, 1024 @ 32 kHz). Only set this to override that.
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

	// UseNeural enables the custom neural network for learned embeddings.
	// Trains a small feedforward net on enrollment data. Default false.
	UseNeural bool

	// NeuralHidden is the hidden layer sizes for the neural network.
	// Default: [128, 64]. Larger = more capacity, slower.
	NeuralHidden []int

	// NeuralEpochs is the number of training epochs. Default 100.
	NeuralEpochs int

	// UseIVector enables i-vector + PLDA scoring.
	// More accurate but slower. Default false.
	UseIVector bool

	// IVDim is the i-vector dimension. Default 128.
	IVDim int

	// IVComponents is the number of UBM components. Default 64.
	IVComponents int

	// UseRASTA enables RASTA filtering for noise robustness. Default false.
	UseRASTA bool

	// UsePLP uses PLP features instead of MFCC. Default false.
	UsePLP bool

	// UseVAD enables voice activity detection. Default false.
	UseVAD bool
}

// DefaultEngineConfig returns default configuration.
func DefaultEngineConfig() EngineConfig {
	return EngineConfig{
		FeatureDim:     39,
		NumMFCC:        13,
		NumFilters:     26,
		MatchThreshold: 0.5,
		TopN:           5,
		TrainMethod:    "centroid",
		GMMComponents:  8,
		NeuralHidden:   []int{128, 64},
		NeuralEpochs:   100,
		IVDim:          128,
		IVComponents:   64,
	}
}

// MatchResult holds the result of speaker identification.
type MatchResult struct {
	Identified  bool
	SpeakerID   string
	SpeakerName string
	Score       float64
	AllMatches  []SpeakerMatch
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
	if len(cfg.NeuralHidden) == 0 {
		cfg.NeuralHidden = []int{128, 64}
	}
	if cfg.NeuralEpochs <= 0 {
		cfg.NeuralEpochs = 100
	}
	if cfg.IVDim <= 0 {
		cfg.IVDim = 128
	}
	if cfg.IVComponents <= 0 {
		cfg.IVComponents = 64
	}
	return &Engine{
		speakers:   make(map[string]*SpeakerModel),
		nnEnrolled: make(map[string][]float64),
		ivEnrolled: make(map[string][]float64),
		dim:        cfg.FeatureDim,
		cfg:        cfg,
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

	// Train neural network and i-vector if enabled (needs >= 2 speakers)
	if len(e.speakers) >= 2 {
		e.trainAdvanced()
	}

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

	// Train neural network and i-vector if enabled (needs >= 2 speakers)
	if len(e.speakers) >= 2 {
		e.trainAdvanced()
	}
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

	// Use advanced scoring (neural + i-vector) if available
	if (e.nn != nil && len(e.nnEnrolled) > 0) || (e.ivModel != nil && len(e.ivEnrolled) > 0) {
		return e.identifyAdvanced(features)
	}

	matches := make([]SpeakerMatch, 0, len(e.speakers))

	for _, model := range e.speakers {
		var score float64
		if e.cfg.TrainMethod == "dtw" {
			score = model.ScoreSequence(features)
		} else {
			// For centroid/gmm, compare against mean of query features
			meanFeatures := matcher.MeanVector(features)
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
	if e.cfg.TopN > 0 && len(matches) > e.cfg.TopN {
		matches = matches[:e.cfg.TopN]
	}

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
		meanFeatures := matcher.MeanVector(features)
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
		meanFeatures := matcher.MeanVector(features)
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
		meanFeatures := matcher.MeanVector(features)
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
			ID:          model.ID,
			Name:        model.Name,
			NumSamples:  model.NumSamples,
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
	// Optional: Voice Activity Detection
	if e.cfg.UseVAD && len(samples) > 0 {
		voiced := dsp.VAD(samples, sampleRate)
		if len(voiced) > 0 {
			// Keep only voiced segments
			var filtered []float64
			frameLen := int(25.0 * float64(sampleRate) / 1000.0)
			frameShift := int(10.0 * float64(sampleRate) / 1000.0)
			for i, v := range voiced {
				if v {
					start := i * frameShift
					end := start + frameLen
					if end > len(samples) {
						end = len(samples)
					}
					filtered = append(filtered, samples[start:end]...)
				}
			}
			if len(filtered) > frameLen {
				samples = filtered
			}
		}
	}

	// Extract MFCC with deltas, honoring the configured feature parameters
	mfccCfg := dsp.DefaultMFCCConfig(sampleRate)
	if e.cfg.NumMFCC > 0 {
		mfccCfg.NumMFCC = e.cfg.NumMFCC
	}
	if e.cfg.NumFilters > 0 {
		mfccCfg.NumFilters = e.cfg.NumFilters
	}
	if e.cfg.FFTSize > 0 {
		mfccCfg.FFTSize = e.cfg.FFTSize
	}
	result := dsp.ExtractMFCCFromRawConfig(samples, sampleRate, mfccCfg)

	// Optional: RASTA filtering of each cepstral coefficient over time
	if e.cfg.UseRASTA && len(result.Coefficients) > 0 {
		numCoeffs := len(result.Coefficients[0])
		for c := 4; c < numCoeffs; c++ {
			cepstrum := make([]float64, len(result.Coefficients))
			for t := range result.Coefficients {
				cepstrum[t] = result.Coefficients[t][c]
			}
			if filtered := dsp.RASTAFilter([][]float64{cepstrum}); len(filtered) > 0 {
				for t := range result.Coefficients {
					result.Coefficients[t][c] = filtered[0][t]
				}
			}
		}
	}

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

	keep := make(map[string]bool, len(e.speakers))
	for _, model := range e.speakers {
		keep[model.ID+".json"] = true
		path := filepath.Join(dir, model.ID+".json")
		if err := saveModel(model, path); err != nil {
			return fmt.Errorf("save speaker %s: %w", model.ID, err)
		}
	}

	// Remove files of speakers that no longer exist, otherwise they would be
	// resurrected by the next LoadDatabase.
	entries, err := os.ReadDir(dir)
	if err != nil {
		return fmt.Errorf("read db dir: %w", err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".json") || keep[name] {
			continue
		}
		if err := os.Remove(filepath.Join(dir, name)); err != nil {
			return fmt.Errorf("remove stale speaker file %s: %w", name, err)
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

	// Rebuild advanced models so a loaded database scores the same way as a
	// freshly enrolled one.
	if len(e.speakers) >= 2 {
		e.trainAdvanced()
	}

	return nil
}

// GetModel returns a speaker model (for advanced usage).
func (e *Engine) GetModel(id string) *SpeakerModel {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.speakers[id]
}

// --- Advanced: Neural + i-vector training ---

// trainAdvanced trains neural network and i-vector models on all enrolled speakers.
func (e *Engine) trainAdvanced() {
	// Collect all features by speaker
	speakerFeats := make(map[string][][]float64)
	for id, model := range e.speakers {
		if len(model.Features) > 0 {
			speakerFeats[id] = model.Features
		}
	}

	// Train neural network
	if e.cfg.UseNeural && len(speakerFeats) >= 2 {
		featDim := e.dim
		if len(e.speakers) > 0 {
			for _, m := range e.speakers {
				if len(m.Features) > 0 && len(m.Features[0]) > 0 {
					featDim = len(m.Features[0])
					break
				}
			}
		}

		e.nn = neural.New(featDim, e.cfg.NeuralHidden, 64, neural.ActLeakyReLU)

		// Train with contrastive loss
		trainCfg := neural.DefaultTrainConfig()
		trainCfg.Epochs = e.cfg.NeuralEpochs
		e.nn.Train(speakerFeats, trainCfg)

		// Compute enrolled embeddings
		e.nnEnrolled = make(map[string][]float64)
		for id, feats := range speakerFeats {
			e.nnEnrolled[id] = e.nn.Embed(feats)
		}
	}

	// Train i-vector + PLDA
	if e.cfg.UseIVector && len(speakerFeats) >= 2 {
		// Flatten all features for UBM training
		var allFeats [][]float64
		for _, feats := range speakerFeats {
			allFeats = append(allFeats, feats...)
		}

		e.ivModel = ivector.NewIVectorModel(e.cfg.IVDim)
		e.ivModel.FitUBM(allFeats, e.cfg.IVComponents)
		e.ivModel.FitT(allFeats)

		// Extract i-vectors for each speaker
		ivBySpeaker := make(map[string][][]float64)
		e.ivEnrolled = make(map[string][]float64)
		for id, feats := range speakerFeats {
			iv := e.ivModel.ExtractIVectorFromSequence(feats)
			e.ivEnrolled[id] = iv
			ivBySpeaker[id] = [][]float64{iv}
		}

		// Train PLDA
		e.plda = ivector.NewPLDAModel(64)
		e.plda.Fit(ivBySpeaker)
	}
}

// identifyAdvanced scores using neural + i-vector, returns best match.
func (e *Engine) identifyAdvanced(features [][]float64) *MatchResult {
	scores := make(map[string]float64)

	hasNN := e.nn != nil && len(e.nnEnrolled) > 0
	hasIV := e.ivModel != nil && e.plda != nil && len(e.ivEnrolled) > 0

	// Weights sum to 1 so that a single active scorer can still reach the
	// match threshold on its own.
	var wNN, wIV float64
	switch {
	case hasNN && hasIV:
		wNN, wIV = 0.6, 0.4
	case hasNN:
		wNN = 1.0
	case hasIV:
		wIV = 1.0
	}

	// Neural network scoring
	if hasNN {
		queryEmb := e.nn.Embed(features)
		for id, enrolledEmb := range e.nnEnrolled {
			scores[id] += matcher.CosineSimilarity(queryEmb, enrolledEmb) * wNN
		}
	}

	// i-vector + PLDA scoring
	if hasIV {
		queryIV := e.ivModel.ExtractIVectorFromSequence(features)
		for id, enrolledIV := range e.ivEnrolled {
			pldaScore := e.plda.Score(queryIV, enrolledIV)
			// Normalize PLDA score to [-1, 1]
			normScore := pldaScore / (1.0 + math.Abs(pldaScore))
			scores[id] += normScore * wIV
		}
	}

	if len(scores) == 0 {
		return &MatchResult{Identified: false}
	}

	// Build sorted matches
	matches := make([]SpeakerMatch, 0, len(scores))
	for id, score := range scores {
		name := id
		if m, ok := e.speakers[id]; ok {
			name = m.Name
		}
		matches = append(matches, SpeakerMatch{
			SpeakerID:   id,
			SpeakerName: name,
			Score:       score,
		})
	}
	sort.Slice(matches, func(i, j int) bool {
		return matches[i].Score > matches[j].Score
	})
	if e.cfg.TopN > 0 && len(matches) > e.cfg.TopN {
		matches = matches[:e.cfg.TopN]
	}

	result := &MatchResult{AllMatches: matches}
	if len(matches) > 0 && matches[0].Score >= e.cfg.MatchThreshold {
		result.Identified = true
		result.SpeakerID = matches[0].SpeakerID
		result.SpeakerName = matches[0].SpeakerName
		result.Score = matches[0].Score
	}

	return result
}

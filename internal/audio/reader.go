package audio

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/gopxl/beep"
	"github.com/gopxl/beep/mp3"
	"github.com/gopxl/beep/wav"
)

// AudioData holds decoded audio samples and metadata.
type AudioData struct {
	Samples    []float64 // mono PCM samples normalized to [-1, 1]
	SampleRate int
	Channels   int
	BitDepth   int
}

// ReadFile reads a WAV or MP3 file and returns normalized mono samples.
func ReadFile(path string) (*AudioData, error) {
	ext := strings.ToLower(filepath.Ext(path))
	switch ext {
	case ".wav":
		return readWAV(path)
	case ".mp3":
		return readMP3(path)
	default:
		return nil, fmt.Errorf("unsupported format: %s (use .wav or .mp3)", ext)
	}
}

func readMP3(path string) (*AudioData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mp3: %w", err)
	}
	defer f.Close()

	streamer, format, err := mp3.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode mp3: %w", err)
	}
	defer streamer.Close()

	return readStreamer(streamer, format)
}

func readWAV(path string) (*AudioData, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open wav: %w", err)
	}
	defer f.Close()

	streamer, format, err := wav.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decode wav: %w", err)
	}
	defer streamer.Close()

	return readStreamer(streamer, format)
}

func readStreamer(streamer beep.Streamer, format beep.Format) (*AudioData, error) {
	sampleRate := format.SampleRate
	channels := format.NumChannels

	var allSamples [][2]float64
	buf := make([][2]float64, 4096)
	for {
		n, ok := streamer.Stream(buf)
		if n > 0 {
			allSamples = append(allSamples, buf[:n]...)
		}
		if !ok {
			break
		}
	}

	samples := make([]float64, len(allSamples))
	for i, s := range allSamples {
		if channels > 1 {
			samples[i] = (s[0] + s[1]) / 2.0
		} else {
			samples[i] = s[0]
		}
	}

	return &AudioData{
		Samples:    samples,
		SampleRate: int(sampleRate),
		Channels:   1,
		BitDepth:   16,
	}, nil
}

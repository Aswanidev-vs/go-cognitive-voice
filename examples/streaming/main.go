// streaming demonstrates real-time speaker identification from a chunked audio source.
//
// It reads a WAV file in chunks (simulating a live stream) and identifies
// speakers as audio accumulates.
//
// Usage:
//
//	go run examples/streaming/main.go audio.wav
package main

import (
	"fmt"
	"os"

	"github.com/Aswanidev-vs/go-cognitive-voice/pkg/gcv"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintf(os.Stderr, "Usage: %s <audio-file>\n", os.Args[0])
		os.Exit(1)
	}

	path := os.Args[1]
	sampleRate := 16000
	chunkSize := sampleRate / 4 // 250ms chunks

	// Load the full audio (in real usage, you'd read from mic/websocket/file)
	samples, sr, err := gcv.ReadAudio(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}
	if sr != sampleRate {
		sampleRate = sr
		chunkSize = sampleRate / 4
	}

	// Create engine and optionally enroll speakers
	engine := gcv.NewEngine(gcv.DefaultConfig())
	// engine.EnrollSpeaker("alice", "Alice", []string{"alice.wav"})

	// Create streaming buffer — identify every 1 second
	buf := engine.NewStreamBuffer(sampleRate, 1.0)

	fmt.Printf("Streaming %s (%.1fs, %d Hz)\n", path, float64(len(samples))/float64(sampleRate), sampleRate)
	fmt.Printf("Chunk size: %d samples (%.0fms)\n\n", chunkSize, float64(chunkSize)/float64(sampleRate)*1000)

	// Simulate streaming by pushing samples in chunks
	totalChunks := (len(samples) + chunkSize - 1) / chunkSize
	identified := 0

	for i := 0; i < totalChunks; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(samples) {
			end = len(samples)
		}

		buf.Write(samples[start:end])

		if result, ok := buf.Identify(); ok {
			identified++
			status := "NO MATCH"
			if result.Identified {
				status = result.SpeakerName
			}
			fmt.Printf("[%.1fs] buffer=%.1fs | %s (score: %.3f)\n",
				buf.Duration()-1.0, buf.Duration(), status, result.Score)
		}
	}

	// Final identification on full audio
	fmt.Println("\n--- Final (full buffer) ---")
	result := buf.IdentifyFull()
	if result.Identified {
		fmt.Printf("Speaker: %s (score: %.3f)\n", result.SpeakerName, result.Score)
	} else {
		fmt.Println("No matching speaker found.")
		for _, m := range result.AllMatches {
			fmt.Printf("  %s: %.4f\n", m.SpeakerName, m.Score)
		}
	}

	fmt.Printf("\nChunks processed: %d, identified: %d\n", totalChunks, identified)
}

// basic demonstrates the simplest speaker identification workflow.
//
// Usage:
//
//	go run examples/basic/main.go
package main

import (
	"fmt"
	"os"

	"github.com/Aswanidev-vs/go-cognitive-voice/pkg/gcv"
)

func main() {
	// 1. Create engine with defaults
	engine := gcv.NewEngine(gcv.DefaultConfig())

	// 2. Enroll speakers — provide any number of audio files per speaker.
	//    More files = better accuracy. Supported: WAV, MP3.
	if err := engine.EnrollSpeaker("alice", "Alice", []string{
		"recordings/alice_01.wav",
		"recordings/alice_02.wav", // optional second recording
	}); err != nil {
		fmt.Fprintf(os.Stderr, "enroll alice: %v\n", err)
		os.Exit(1)
	}

	if err := engine.EnrollSpeaker("bob", "Bob", []string{
		"recordings/bob_01.wav",
	}); err != nil {
		fmt.Fprintf(os.Stderr, "enroll bob: %v\n", err)
		os.Exit(1)
	}

	// 3. Identify an unknown voice
	result, err := engine.Identify("recordings/mystery.wav")
	if err != nil {
		fmt.Fprintf(os.Stderr, "identify: %v\n", err)
		os.Exit(1)
	}

	if result.Identified {
		fmt.Printf("Speaker: %s (confidence: %.1f%%)\n", result.SpeakerName, result.Score*100)
	} else {
		fmt.Println("No matching speaker found.")
	}

	// 4. Show all candidates ranked by score
	fmt.Println("\nAll matches:")
	for _, m := range result.AllMatches {
		fmt.Printf("  %-10s score: %.4f\n", m.SpeakerName, m.Score)
	}
}

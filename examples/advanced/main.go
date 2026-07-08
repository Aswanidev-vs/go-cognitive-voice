// advanced demonstrates persistence, threshold tuning, and speaker verification.
//
// Usage:
//
//	go run examples/advanced/main.go
package main

import (
	"fmt"
	"os"

	"github.com/Aswanidev-vs/go-cognitive-voice/pkg/gcv"
)

const dbDir = "./speaker_db"

func main() {
	cfg := gcv.DefaultConfig()
	cfg.MatchThreshold = 0.85 // stricter than default 0.5

	engine := gcv.NewEngine(cfg)

	// Try loading existing database first
	if err := engine.LoadDatabase(dbDir); err != nil {
		fmt.Printf("No existing database, starting fresh.\n")
	}

	// Enroll speakers
	enroll := func(id, name string, files []string) {
		if err := engine.EnrollSpeaker(id, name, files); err != nil {
			fmt.Fprintf(os.Stderr, "enroll %s: %v\n", name, err)
			os.Exit(1)
		}
		fmt.Printf("Enrolled: %s (%s)\n", name, id)
	}

	enroll("mai_jp", "Mai (JP)", []string{"mai_san_v2.wav"})
	enroll("mai_en", "Mai (EN)", []string{"mai_san_v3.wav"})

	// Save database for next run
	if err := engine.SaveDatabase(dbDir); err != nil {
		fmt.Fprintf(os.Stderr, "save: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Database saved to %s\n\n", dbDir)

	// --- Identify ---
	fmt.Println("=== Identification ===")
	for _, file := range []string{"mai_san_v2.wav", "mai_san_v3.wav"} {
		result, err := engine.Identify(file)
		if err != nil {
			fmt.Fprintf(os.Stderr, "identify %s: %v\n", file, err)
			continue
		}
		if result.Identified {
			fmt.Printf("  %s → %s (%.1f%%)\n", file, result.SpeakerName, result.Score*100)
		} else {
			fmt.Printf("  %s → no match (best: %s at %.1f%%)\n",
				file, result.AllMatches[0].SpeakerName, result.AllMatches[0].Score*100)
		}
	}

	// --- Detect (verify specific speaker) ---
	fmt.Println("\n=== Verification (threshold: 0.85) ===")
	for _, file := range []string{"mai_san_v2.wav", "mai_san_v3.wav"} {
		for _, speakerID := range []string{"mai_jp", "mai_en"} {
			matched, score, _ := engine.Detect(file, speakerID)
			status := "PASS"
			if !matched {
				status = "FAIL"
			}
			fmt.Printf("  %s → %s: %s (%.4f)\n", file, speakerID, status, score)
		}
	}

	// --- List ---
	fmt.Println("\n=== Enrolled Speakers ===")
	for _, s := range engine.ListSpeakers() {
		fmt.Printf("  %-10s %-12s %d samples\n", s.ID, s.Name, s.NumSamples)
	}
}

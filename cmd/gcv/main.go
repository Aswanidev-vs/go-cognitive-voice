// Command gcv is a speaker detection and identification CLI tool.
//
// Usage:
//
//	gcv enroll <speaker-id> <name> <file1.wav> [file2.mp3 ...]
//	gcv identify <audio-file>
//	gcv detect <audio-file> <speaker-id>
//	gcv list
//	gcv remove <speaker-id>
//	gcv features <audio-file>
//	gcv help
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"

	"github.com/Aswanidev-vs/go-cognitive-voice/pkg/gcv"
)

const dbDir = ".gcv-db"

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	cmd := os.Args[1]

	switch cmd {
	case "enroll":
		cmdEnroll()
	case "identify":
		cmdIdentify()
	case "detect":
		cmdDetect()
	case "list":
		cmdList()
	case "remove":
		cmdRemove()
	case "features":
		cmdFeatures()
	case "help", "--help", "-h":
		printUsage()
	case "version", "--version", "-v":
		fmt.Printf("gcv v%s\n", gcv.Version())
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", cmd)
		printUsage()
		os.Exit(1)
	}
}

func cmdEnroll() {
	if len(os.Args) < 5 {
		fmt.Fprintf(os.Stderr, "Usage: gcv enroll <speaker-id> <name> <file1> [file2 ...]\n")
		os.Exit(1)
	}

	speakerID := os.Args[2]
	name := os.Args[3]
	files := os.Args[4:]

	// Validate files
	for _, f := range files {
		if _, err := os.Stat(f); err != nil {
			fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", f)
			os.Exit(1)
		}
	}

	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	// Load existing database
	_ = engine.LoadDatabase(dbDir)

	fmt.Printf("Enrolling speaker '%s' (ID: %s) from %d file(s)...\n", name, speakerID, len(files))

	if err := engine.EnrollSpeaker(speakerID, name, files); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Save database
	if err := engine.SaveDatabase(dbDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error saving database: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Speaker '%s' enrolled successfully.\n", name)
}

func cmdIdentify() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: gcv identify <audio-file>\n")
		os.Exit(1)
	}

	audioPath := os.Args[2]
	if _, err := os.Stat(audioPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", audioPath)
		os.Exit(1)
	}

	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	if err := engine.LoadDatabase(dbDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading database: %v\n", err)
		os.Exit(1)
	}

	speakers := engine.ListSpeakers()
	if len(speakers) == 0 {
		fmt.Println("No speakers enrolled. Use 'gcv enroll' first.")
		os.Exit(0)
	}

	fmt.Printf("Identifying speaker in '%s'...\n", audioPath)
	fmt.Printf("Database: %d enrolled speaker(s)\n\n", len(speakers))

	result, err := engine.Identify(audioPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if result.Identified {
		fmt.Printf("Identified: %s (ID: %s, score: %.4f)\n", result.SpeakerName, result.SpeakerID, result.Score)
	} else {
		fmt.Println("No matching speaker found (below threshold).")
	}

	if len(result.AllMatches) > 0 {
		fmt.Println("\nAll matches:")
		w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
		fmt.Fprintln(w, "  Speaker\tID\tScore")
		fmt.Fprintln(w, "  -------\t--\t-----")
		for _, m := range result.AllMatches {
			fmt.Fprintf(w, "  %s\t%s\t%.4f\n", m.SpeakerName, m.SpeakerID, m.Score)
		}
		w.Flush()
	}
}

func cmdDetect() {
	if len(os.Args) < 4 {
		fmt.Fprintf(os.Stderr, "Usage: gcv detect <audio-file> <speaker-id>\n")
		os.Exit(1)
	}

	audioPath := os.Args[2]
	speakerID := os.Args[3]

	if _, err := os.Stat(audioPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", audioPath)
		os.Exit(1)
	}

	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	if err := engine.LoadDatabase(dbDir); err != nil {
		fmt.Fprintf(os.Stderr, "Error loading database: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Checking if '%s' is speaker '%s'...\n", audioPath, speakerID)

	matched, score, err := engine.Detect(audioPath, speakerID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if matched {
		fmt.Printf("MATCH (score: %.4f)\n", score)
	} else {
		fmt.Printf("NO MATCH (score: %.4f)\n", score)
	}
}

func cmdList() {
	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	_ = engine.LoadDatabase(dbDir)

	speakers := engine.ListSpeakers()
	if len(speakers) == 0 {
		fmt.Println("No speakers enrolled.")
		return
	}

	fmt.Printf("Enrolled speakers (%d):\n\n", len(speakers))
	w := tabwriter.NewWriter(os.Stdout, 0, 0, 2, ' ', 0)
	fmt.Fprintln(w, "  ID\tName\tSamples\tMethod")
	fmt.Fprintln(w, "  --\t----\t-------\t------")
	for _, s := range speakers {
		fmt.Fprintf(w, "  %s\t%s\t%d\t%s\n", s.ID, s.Name, s.NumSamples, s.TrainMethod)
	}
	w.Flush()
}

func cmdRemove() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: gcv remove <speaker-id>\n")
		os.Exit(1)
	}

	speakerID := os.Args[2]

	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	_ = engine.LoadDatabase(dbDir)

	if engine.RemoveSpeaker(speakerID) {
		_ = engine.SaveDatabase(dbDir)
		fmt.Printf("Speaker '%s' removed.\n", speakerID)
	} else {
		fmt.Printf("Speaker '%s' not found.\n", speakerID)
	}
}

func cmdFeatures() {
	if len(os.Args) < 3 {
		fmt.Fprintf(os.Stderr, "Usage: gcv features <audio-file>\n")
		os.Exit(1)
	}

	audioPath := os.Args[2]
	if _, err := os.Stat(audioPath); err != nil {
		fmt.Fprintf(os.Stderr, "Error: file not found: %s\n", audioPath)
		os.Exit(1)
	}

	cfg := gcv.DefaultConfig()
	engine := gcv.NewEngine(cfg)

	features, err := engine.ExtractFeatures(audioPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	ext := strings.ToLower(filepath.Ext(audioPath))
	fmt.Printf("File: %s (%s)\n", audioPath, ext)
	fmt.Printf("Feature frames: %d\n", len(features))
	if len(features) > 0 {
		fmt.Printf("Feature dimension: %d\n", len(features[0]))
		fmt.Printf("\nFirst 5 frames:\n")
		for i := 0; i < 5 && i < len(features); i++ {
			fmt.Printf("  [%d]: ", i)
			for j, v := range features[i] {
				if j > 0 {
					fmt.Print(", ")
				}
				fmt.Printf("%.4f", v)
			}
			fmt.Println()
		}
	}
}

func printUsage() {
	fmt.Println(`gcv - Speaker Detection & Identification (Pure Go)

Usage:
  gcv <command> [arguments]

Commands:
  enroll <id> <name> <file1> [file2 ...]   Enroll a speaker from audio files
  identify <audio-file>                      Identify the speaker in an audio file
  detect <audio-file> <speaker-id>           Check if audio matches a speaker
  list                                       List all enrolled speakers
  remove <speaker-id>                        Remove a speaker
  features <audio-file>                      Show extracted features
  help                                       Show this help message
  version                                    Show version

Supported formats: WAV (PCM 8/16/24/32-bit, float32/64), MP3

Examples:
  gcv enroll alice1 "Alice" recordings/alice_*.wav
  gcv identify mystery_voice.mp3
  gcv detect alice_sample.wav alice1`)
}

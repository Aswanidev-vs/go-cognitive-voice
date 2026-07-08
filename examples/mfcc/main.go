// mfcc demonstrates MFCC feature extraction without speaker identification.
//
// This is useful when you want to build your own matching logic or
// feed features into a different ML pipeline.
//
// Usage:
//
//	go run examples/mfcc/main.go audio.wav
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

	// Read raw audio
	samples, sampleRate, err := gcv.ReadAudio(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "read: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("File: %s\n", path)
	fmt.Printf("Sample rate: %d Hz\n", sampleRate)
	fmt.Printf("Duration: %.2f seconds\n", float64(len(samples))/float64(sampleRate))
	fmt.Printf("Samples: %d\n\n", len(samples))

	// Extract MFCC features
	features := gcv.ExtractMFCC(samples, sampleRate)

	fmt.Printf("Feature frames: %d\n", len(features))
	if len(features) > 0 {
		fmt.Printf("Feature dimension: %d\n\n", len(features[0]))

		fmt.Println("Feature layout:")
		fmt.Println("  [0:14]  = MFCC coefficients (including log energy)")
		fmt.Println("  [14:28] = delta features (1st derivative)")
		fmt.Println("  [28:42] = delta-delta features (2nd derivative)")
		fmt.Println()

		// Show first 3 frames
		fmt.Println("First 3 frames:")
		for i := 0; i < 3 && i < len(features); i++ {
			fmt.Printf("  Frame %d: [", i)
			for j, v := range features[i] {
				if j > 0 {
					fmt.Print(", ")
				}
				fmt.Printf("%.2f", v)
			}
			fmt.Println("]")
		}
	}
}

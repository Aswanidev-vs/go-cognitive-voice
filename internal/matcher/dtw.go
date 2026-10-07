package matcher

import (
	"math"
)

// DTWMode determines how the DTW distance is normalized.
type DTWMode int

const (
	// DTWModeNone returns the raw accumulated distance.
	DTWModeNone DTWMode = iota
	// DTWModeNormalized divides by the path length.
	DTWModeNormalized
	// DTWModePadded divides by the sum of both sequence lengths.
	DTWModePadded
)

// DTWConfig holds DTW computation parameters.
type DTWConfig struct {
	Mode   DTWMode
	Window int  // Sakoe-Chiba band width (0 = no constraint)
	UseLog bool // Use log-distance instead of Euclidean
}

// DefaultDTWConfig returns default parameters.
func DefaultDTWConfig() DTWConfig {
	return DTWConfig{
		Mode:   DTWModeNormalized,
		Window: 0,
		UseLog: false,
	}
}

// DTWResult holds the DTW computation result.
type DTWResult struct {
	Distance float64
	PathLen  int
	Path     [][2]int // Warping path (optional, computed if needed)
}

// DTW computes Dynamic Time Warping distance between two feature sequences.
// Each sequence is [numFrames][numFeatures].
func DTW(seq1, seq2 [][]float64, cfg DTWConfig) DTWResult {
	n := len(seq1)
	m := len(seq2)

	if n == 0 || m == 0 {
		return DTWResult{Distance: math.Inf(1)}
	}

	// Create cost matrix
	cost := make([][]float64, n+1)
	for i := range cost {
		cost[i] = make([]float64, m+1)
		for j := range cost[i] {
			cost[i][j] = math.Inf(1)
		}
	}
	cost[0][0] = 0

	// Fill cost matrix
	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			d := featureDistance(seq1[i-1], seq2[j-1], cfg.UseLog)

			// Sakoe-Chiba band constraint
			if cfg.Window > 0 && int(math.Abs(float64(i-j))) > cfg.Window {
				continue
			}

			cost[i][j] = d + min3(
				cost[i-1][j],   // insertion
				cost[i][j-1],   // deletion
				cost[i-1][j-1], // match
			)
		}
	}

	dist := cost[n][m]
	pathLen := n + m // approximate path length

	switch cfg.Mode {
	case DTWModeNormalized:
		if pathLen > 0 {
			dist /= float64(pathLen)
		}
	case DTWModePadded:
		total := n + m
		if total > 0 {
			dist /= float64(total)
		}
	}

	return DTWResult{
		Distance: dist,
		PathLen:  pathLen,
	}
}

// DTWWithPath computes DTW and returns the optimal warping path.
func DTWWithPath(seq1, seq2 [][]float64, cfg DTWConfig) DTWResult {
	n := len(seq1)
	m := len(seq2)

	if n == 0 || m == 0 {
		return DTWResult{Distance: math.Inf(1)}
	}

	cost := make([][]float64, n+1)
	parent := make([][][2]int, n+1)
	for i := range cost {
		cost[i] = make([]float64, m+1)
		parent[i] = make([][2]int, m+1)
		for j := range cost[i] {
			cost[i][j] = math.Inf(1)
		}
	}
	cost[0][0] = 0

	for i := 1; i <= n; i++ {
		for j := 1; j <= m; j++ {
			d := featureDistance(seq1[i-1], seq2[j-1], cfg.UseLog)

			if cfg.Window > 0 && int(math.Abs(float64(i-j))) > cfg.Window {
				continue
			}

			insertion := cost[i-1][j]
			deletion := cost[i][j-1]
			match := cost[i-1][j-1]

			best := match
			parent[i][j] = [2]int{i - 1, j - 1}

			if insertion < best {
				best = insertion
				parent[i][j] = [2]int{i - 1, j}
			}
			if deletion < best {
				best = deletion
				parent[i][j] = [2]int{i, j - 1}
			}

			cost[i][j] = d + best
		}
	}

	// Traceback path
	var path [][2]int
	i, j := n, m
	for i > 0 || j > 0 {
		path = append(path, [2]int{i - 1, j - 1})
		prev := parent[i][j]
		i, j = prev[0], prev[1]
	}

	// Reverse path
	for left, right := 0, len(path)-1; left < right; left, right = left+1, right-1 {
		path[left], path[right] = path[right], path[left]
	}

	dist := cost[n][m]
	pathLen := len(path)

	switch cfg.Mode {
	case DTWModeNormalized:
		if pathLen > 0 {
			dist /= float64(pathLen)
		}
	case DTWModePadded:
		total := n + m
		if total > 0 {
			dist /= float64(total)
		}
	}

	return DTWResult{
		Distance: dist,
		PathLen:  pathLen,
		Path:     path,
	}
}

// featureDistance computes distance between two feature vectors.
// Uses the shorter vector length; extra elements in the longer vector are ignored.
func featureDistance(a, b []float64, useLog bool) float64 {
	sum := 0.0
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	if n == 0 {
		return 0
	}
	for i := 0; i < n; i++ {
		d := a[i] - b[i]
		sum += d * d
	}
	dist := math.Sqrt(sum)
	if useLog && dist > 0 {
		return math.Log1p(dist)
	}
	return dist
}

func min3(a, b, c float64) float64 {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

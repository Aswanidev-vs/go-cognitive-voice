package neural

import (
	"math"
	"math/rand"
)

// TrainConfig holds training hyperparameters.
type TrainConfig struct {
	Epochs    int     // training epochs (default: 100)
	BatchSize int     // samples per mini-batch (default: 32)
	LR        float64 // learning rate (default: 0.001)
	Margin    float64 // contrastive loss margin (default: 1.0)
}

// DefaultTrainConfig returns sensible defaults.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Epochs:    100,
		BatchSize: 32,
		LR:        0.001,
		Margin:    1.0,
	}
}

// Train trains the network using contrastive loss on speaker features.
// speakerFeatures is a map of speaker ID -> list of feature vectors.
// The network learns to produce similar embeddings for same-speaker pairs
// and different embeddings for different-speaker pairs.
func (n *Network) Train(speakerFeatures map[string][][]float64, cfg TrainConfig) float64 {
	// Build pairs: (anchor, positive) for same speaker, (anchor, negative) for different
	type pair struct {
		a, b []float64
		same bool
	}

	var pairs []pair
	ids := make([]string, 0, len(speakerFeatures))
	for id := range speakerFeatures {
		ids = append(ids, id)
	}

	if len(ids) < 2 {
		// Can't do contrastive learning with 1 speaker
		// Just use self-reconstruction
		return 0
	}

	// Generate same-speaker pairs
	for _, feats := range speakerFeatures {
		for i := 0; i < len(feats); i++ {
			for j := i + 1; j < len(feats) && j < i+10; j++ {
				pairs = append(pairs, pair{feats[i], feats[j], true})
			}
		}
	}

	// Generate different-speaker pairs
	for i := 0; i < len(ids); i++ {
		for j := i + 1; j < len(ids); j++ {
			featsA := speakerFeatures[ids[i]]
			featsB := speakerFeatures[ids[j]]
			// Take up to 5 pairs per speaker combination
			nPairs := 5
			if len(featsA) < nPairs {
				nPairs = len(featsA)
			}
			if len(featsB) < nPairs {
				nPairs = len(featsB)
			}
			for k := 0; k < nPairs; k++ {
				aIdx := rand.Intn(len(featsA))
				bIdx := rand.Intn(len(featsB))
				pairs = append(pairs, pair{featsA[aIdx], featsB[bIdx], false})
			}
		}
	}

	if len(pairs) == 0 {
		return 0
	}

	if cfg.BatchSize <= 0 {
		cfg.BatchSize = DefaultTrainConfig().BatchSize
	}

	// Shuffle pairs
	rand.Shuffle(len(pairs), func(i, j int) {
		pairs[i], pairs[j] = pairs[j], pairs[i]
	})

	bestLoss := math.Inf(1)

	for epoch := 0; epoch < cfg.Epochs; epoch++ {
		totalLoss := 0.0

		for start := 0; start < len(pairs); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(pairs) {
				end = len(pairs)
			}
			// Forward + backward for each pair in the batch
			for _, p := range pairs[start:end] {
				totalLoss += n.trainPair(p.a, p.b, p.same, cfg)
			}
		}

		avgLoss := totalLoss / float64(len(pairs))
		if avgLoss < bestLoss {
			bestLoss = avgLoss
		}

		// Decay learning rate
		if epoch > 0 && epoch%30 == 0 {
			cfg.LR *= 0.5
		}
	}

	return bestLoss
}

// trainPair runs forward and backward passes for one contrastive pair and
// returns its loss.
//
// Same-speaker pairs minimize the squared cosine distance; different-speaker
// pairs push the embeddings apart until the margin is reached.
func (n *Network) trainPair(a, b []float64, same bool, cfg TrainConfig) float64 {
	embA, cacheA := n.forwardCached(a)
	embB, cacheB := n.forwardCached(b)
	dist := cosineDistance(embA, embB)

	var scale, loss float64
	if same {
		loss = dist * dist
		scale = 2 * dist // d(dist²)/d(dist)
	} else {
		loss = math.Max(0, cfg.Margin-dist)
		if loss == 0 {
			return 0
		}
		scale = -1 // d(margin - dist)/d(dist)
	}

	gradA, gradB := cosineDistanceGrads(embA, embB)
	for i := range gradA {
		gradA[i] *= scale
		gradB[i] *= scale
	}
	n.backprop(cacheA, gradA, cfg.LR)
	n.backprop(cacheB, gradB, cfg.LR)
	return loss
}

// cosineDistanceGrads returns the gradients of the cosine distance
// (1 - cos(a, b)) with respect to a and b.
func cosineDistanceGrads(a, b []float64) ([]float64, []float64) {
	gradA := make([]float64, len(a))
	gradB := make([]float64, len(b))

	normA := vectorNorm(a)
	normB := vectorNorm(b)
	if normA < 1e-10 || normB < 1e-10 {
		return gradA, gradB
	}

	var dot float64
	for i := range a {
		dot += a[i] * b[i]
	}
	c := dot / (normA * normB)

	// d(1-c)/da = -(b̂ - c·â)/|a|, symmetric for b
	for i := range a {
		ua, ub := a[i]/normA, b[i]/normB
		gradA[i] = -(ub - c*ua) / normA
		gradB[i] = -(ua - c*ub) / normB
	}
	return gradA, gradB
}

func vectorNorm(v []float64) float64 {
	sum := 0.0
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

// Predict returns the speaker ID with the highest similarity.
func (n *Network) Predict(features [][]float64, enrolled map[string][]float64) (string, float64) {
	emb := n.Embed(features)

	bestID := ""
	bestScore := -1.0

	for id, enrolledEmb := range enrolled {
		sim := cosineSimilarity(emb, enrolledEmb)
		if sim > bestScore {
			bestScore = sim
			bestID = id
		}
	}

	return bestID, bestScore
}

func cosineDistance(a, b []float64) float64 {
	return 1.0 - cosineSimilarity(a, b)
}

func cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom < 1e-10 {
		return 0
	}
	return dot / denom
}

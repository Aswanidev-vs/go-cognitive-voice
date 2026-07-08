package matcher

import "math"

// CosineSimilarity computes cosine similarity between two vectors.
// Returns a value in [-1, 1] where 1 means identical direction.
func CosineSimilarity(a, b []float64) float64 {
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
	if denom < 1e-20 {
		return 0
	}
	return dot / denom
}

// CosineDistance returns 1 - CosineSimilarity, in [0, 2].
func CosineDistance(a, b []float64) float64 {
	return 1.0 - CosineSimilarity(a, b)
}

// EuclideanDistance computes Euclidean distance between two vectors.
func EuclideanDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	sum := 0.0
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return math.Sqrt(sum)
}

// MahalanobisDistance computes Mahalanobis distance between two vectors
// given a shared covariance matrix (or its inverse).
// If invCov is nil, falls back to Euclidean distance.
func MahalanobisDistance(a, b, invCov []float64, dim int) float64 {
	if invCov == nil || len(invCov) != dim*dim {
		return EuclideanDistance(a, b)
	}

	diff := make([]float64, dim)
	for i := 0; i < dim; i++ {
		diff[i] = a[i] - b[i]
	}

	// Compute diff^T * invCov * diff
	sum := 0.0
	for i := 0; i < dim; i++ {
		tmp := 0.0
		for j := 0; j < dim; j++ {
			tmp += invCov[i*dim+j] * diff[j]
		}
		sum += diff[i] * tmp
	}

	if sum < 0 {
		sum = 0
	}
	return math.Sqrt(sum)
}

// MeanVector computes the element-wise mean of a set of vectors.
func MeanVector(vectors [][]float64) []float64 {
	if len(vectors) == 0 {
		return nil
	}
	dim := len(vectors[0])
	mean := make([]float64, dim)
	for _, v := range vectors {
		for i := 0; i < dim && i < len(v); i++ {
			mean[i] += v[i]
		}
	}
	n := float64(len(vectors))
	for i := range mean {
		mean[i] /= n
	}
	return mean
}

// CovarianceMatrix computes the covariance matrix of a set of vectors.
func CovarianceMatrix(vectors [][]float64) [][]float64 {
	n := len(vectors)
	if n == 0 {
		return nil
	}
	dim := len(vectors[0])
	mean := MeanVector(vectors)

	cov := make([][]float64, dim)
	for i := range cov {
		cov[i] = make([]float64, dim)
	}

	for _, v := range vectors {
		diff := make([]float64, dim)
		for d := 0; d < dim; d++ {
			diff[d] = v[d] - mean[d]
		}
		for i := 0; i < dim; i++ {
			for j := 0; j < dim; j++ {
				cov[i][j] += diff[i] * diff[j]
			}
		}
	}

	scale := 1.0 / float64(n-1)
	for i := range cov {
		for j := range cov[i] {
			cov[i][j] *= scale
		}
	}

	return cov
}

// NormalizeVector normalizes a vector to unit length (L2 norm).
func NormalizeVector(v []float64) []float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm < 1e-20 {
		out := make([]float64, len(v))
		copy(out, v)
		return out
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

// Standardize normalizes features to zero mean, unit variance.
func Standardize(vectors [][]float64) [][]float64 {
	if len(vectors) == 0 {
		return nil
	}
	dim := len(vectors[0])
	n := float64(len(vectors))

	mean := make([]float64, dim)
	stddev := make([]float64, dim)

	// Compute mean
	for _, v := range vectors {
		for i := 0; i < dim && i < len(v); i++ {
			mean[i] += v[i]
		}
	}
	for i := range mean {
		mean[i] /= n
	}

	// Compute stddev
	for _, v := range vectors {
		for i := 0; i < dim && i < len(v); i++ {
			d := v[i] - mean[i]
			stddev[i] += d * d
		}
	}
	for i := range stddev {
		stddev[i] = math.Sqrt(stddev[i] / n)
		if stddev[i] < 1e-20 {
			stddev[i] = 1.0
		}
	}

	// Standardize
	result := make([][]float64, len(vectors))
	for idx, v := range vectors {
		result[idx] = make([]float64, dim)
		for i := 0; i < dim && i < len(v); i++ {
			result[idx][i] = (v[i] - mean[i]) / stddev[i]
		}
	}

	return result
}

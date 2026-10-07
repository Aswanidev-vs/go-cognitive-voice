package speaker

import (
	"math"
	"math/rand"
	"sort"

	"github.com/Aswanidev-vs/go-cognitive-voice/internal/matcher"
)

// SpeakerModel represents a trained speaker model.
type SpeakerModel struct {
	ID          string
	Name        string
	Centroid    []float64   // Mean feature vector (centroid approach)
	Covariance  [][]float64 // Covariance matrix
	InvCovFlat  []float64   // Flattened inverse covariance
	Dim         int         // Feature dimension
	NumSamples  int         // Number of enrollment samples
	Features    [][]float64 // All enrollment features (for DTW)
	TrainMethod string      // "centroid", "gmm", or "dtw"
	GMM         *GMMModel   // GMM model (for GMM method)
}

// GMMModel is an alias for GMM.
type GMMModel = GMM

// NewSpeakerModel creates a new empty speaker model.
func NewSpeakerModel(id, name string, dim int) *SpeakerModel {
	return &SpeakerModel{
		ID:   id,
		Name: name,
		Dim:  dim,
	}
}

// TrainCentroid trains a centroid-based speaker model.
// This is the simplest approach: compute the mean feature vector.
func (m *SpeakerModel) TrainCentroid(features [][]float64) {
	if len(features) > 0 && len(features[0]) > 0 {
		m.Dim = len(features[0])
	}
	m.Centroid = matcher.MeanVector(features)
	m.Covariance = matcher.CovarianceMatrix(features)
	m.Features = features
	m.NumSamples = len(features)
	m.TrainMethod = "centroid"

	// Compute inverse covariance for Mahalanobis distance
	if len(m.Covariance) > 0 {
		dim := len(m.Covariance)
		m.InvCovFlat = invertMatrix(flatMatrix(m.Covariance), dim)
	}
}

// TrainDTW trains a DTW-based model (stores all enrollment features).
func (m *SpeakerModel) TrainDTW(features [][]float64) {
	if len(features) > 0 && len(features[0]) > 0 {
		m.Dim = len(features[0])
	}
	m.Features = features
	m.NumSamples = len(features)
	m.TrainMethod = "dtw"

	// Also compute centroid for fallback
	if len(features) > 0 {
		m.Centroid = matcher.MeanVector(features)
	}
}

// TrainGMM trains a Gaussian Mixture Model for the speaker.
func (m *SpeakerModel) TrainGMM(features [][]float64, k int, maxIter int, tol float64) {
	// Derive the dimension from the data: the configured feature dimension is
	// only a fallback and can differ from what extraction actually produces
	// (e.g. 39 configured vs 42 actual = (13 MFCC + energy) * 3 deltas).
	if len(features) > 0 && len(features[0]) > 0 {
		m.Dim = len(features[0])
	}
	if k <= 0 {
		k = min(8, len(features)/5)
		if k < 2 {
			k = 2
		}
	}

	gmm := fitGMM(features, k, m.Dim, maxIter, tol)

	m.GMM = gmm
	m.Features = features
	m.NumSamples = len(features)
	m.TrainMethod = "gmm"
	if len(features) > 0 {
		m.Centroid = matcher.MeanVector(features)
	}
}

// Score computes a match score between the model and query features.
// Higher score = better match. Range depends on method.
func (m *SpeakerModel) Score(query []float64) float64 {
	switch m.TrainMethod {
	case "centroid":
		return m.scoreCentroid(query)
	case "gmm":
		return m.scoreGMM(query)
	case "dtw":
		return 0 // DTW requires full sequence, use ScoreSequence
	default:
		return 0
	}
}

// ScoreSequence computes a match score for a sequence of features (DTW-based).
func (m *SpeakerModel) ScoreSequence(query [][]float64) float64 {
	if m.TrainMethod == "dtw" && len(m.Features) > 0 {
		cfg := matcher.DefaultDTWConfig()
		result := matcher.DTW(m.Features, query, cfg)
		// Convert distance to similarity score (lower distance = higher score)
		if result.Distance <= 0 {
			return 1.0
		}
		return 1.0 / (1.0 + result.Distance)
	}

	// Fallback: compare centroids of the sequence
	if len(query) > 0 && m.Centroid != nil {
		qCentroid := matcher.MeanVector(query)
		return matcher.CosineSimilarity(m.Centroid, qCentroid)
	}

	return 0
}

func (m *SpeakerModel) scoreCentroid(query []float64) float64 {
	if m.Centroid == nil {
		return 0
	}

	// Use cosine similarity as primary score
	sim := matcher.CosineSimilarity(m.Centroid, query)

	// Optionally boost with Mahalanobis distance
	if m.InvCovFlat != nil {
		dim := len(m.Centroid)
		mahal := matcher.MahalanobisDistance(query, m.Centroid, m.InvCovFlat, dim)
		// Convert Mahalanobis to a score (lower distance = higher score)
		mahalScore := 1.0 / (1.0 + mahal)
		// Weighted combination
		return 0.7*sim + 0.3*mahalScore
	}

	return sim
}

func (m *SpeakerModel) scoreGMM(query []float64) float64 {
	if m.GMM == nil {
		return m.scoreCentroid(query)
	}

	// Compute log-likelihood under the GMM
	logLik := gmmLogLikelihood(m.GMM, query)

	// Convert to a similarity score
	// Normalize by dimension for fair comparison across different feature sizes
	score := logLik / float64(len(query))

	// Sigmoid to map to [0, 1]
	return 1.0 / (1.0 + math.Exp(-score))
}

// GMM holds the GMM model (exported for serialization).
type GMM struct {
	Weights []float64
	Means   [][]float64
	Covars  [][][]float64
	InvCovs [][]float64 // Flattened inverse covariances [K][dim*dim]
	K       int
	Dim     int
}

// fitGMM fits a GMM using the EM algorithm.
func fitGMM(data [][]float64, k, dim, maxIter int, tol float64) *GMM {
	n := len(data)
	if n == 0 || k == 0 {
		return nil
	}
	// Clamp k to data size — can't have more components than data points
	if k > n {
		k = n
	}

	gmm := &GMM{
		K:       k,
		Dim:     dim,
		Weights: make([]float64, k),
		Means:   make([][]float64, k),
		Covars:  make([][][]float64, k),
		InvCovs: make([][]float64, k),
	}

	// Initialize with K-means++ style
	centers := kmeansPPInit(data, k)
	for i := 0; i < k; i++ {
		gmm.Means[i] = make([]float64, dim)
		copy(gmm.Means[i], centers[i])
		gmm.Covars[i] = identityMatrix(dim)
		gmm.Weights[i] = 1.0 / float64(k)
	}

	// Responsibility matrix
	gamma := make([][]float64, n)
	for i := range gamma {
		gamma[i] = make([]float64, k)
	}

	for iter := 0; iter < maxIter; iter++ {
		// Sample-independent quantities, valid until the next M-step
		invCovs, logDets := prepareGMM(gmm)
		oldLogLik := gmmLogLikelihoodPrepared(gmm, data, invCovs, logDets)

		// E-step: responsibilities in the log domain (avoids underflow that
		// would silently drop far-away samples)
		for i := 0; i < n; i++ {
			maxLog := math.Inf(-1)
			for j := 0; j < k; j++ {
				gamma[i][j] = math.Log(gmm.Weights[j]) +
					gaussianLogPDFPrepared(data[i], gmm.Means[j], invCovs[j], logDets[j])
				if gamma[i][j] > maxLog {
					maxLog = gamma[i][j]
				}
			}
			if math.IsInf(maxLog, -1) {
				for j := range gamma[i] {
					gamma[i][j] = 0
				}
				continue
			}
			total := 0.0
			for j := 0; j < k; j++ {
				gamma[i][j] = math.Exp(gamma[i][j] - maxLog)
				total += gamma[i][j]
			}
			if total > 0 {
				for j := range gamma[i] {
					gamma[i][j] /= total
				}
			}
		}

		// M-step: update parameters
		for j := 0; j < k; j++ {
			// Effective number of samples for component j
			Nj := 0.0
			for i := 0; i < n; i++ {
				Nj += gamma[i][j]
			}

			if Nj < 1e-10 {
				continue
			}

			// Update weight
			gmm.Weights[j] = Nj / float64(n)

			// Update mean
			for d := 0; d < dim; d++ {
				sum := 0.0
				for i := 0; i < n; i++ {
					if d < len(data[i]) {
						sum += gamma[i][j] * data[i][d]
					}
				}
				gmm.Means[j][d] = sum / Nj
			}

			// Update covariance
			for d1 := 0; d1 < dim; d1++ {
				for d2 := 0; d2 < dim; d2++ {
					sum := 0.0
					for i := 0; i < n; i++ {
						v1 := data[i][d1] - gmm.Means[j][d1]
						v2 := data[i][d2] - gmm.Means[j][d2]
						sum += gamma[i][j] * v1 * v2
					}
					gmm.Covars[j][d1][d2] = sum / Nj
				}
			}

			// Regularize to avoid singular covariance, and shrink toward a
			// scaled identity: components usually have far fewer samples than
			// dimensions, so the raw estimate is rank-deficient and its
			// inverse would explode along the null space.
			regularizeCovariance(gmm.Covars[j], dim, 1e-6)
			avgVar := 0.0
			for d := 0; d < dim; d++ {
				avgVar += gmm.Covars[j][d][d]
			}
			avgVar /= float64(dim)
			for d := 0; d < dim; d++ {
				gmm.Covars[j][d][d] += avgVar
			}

			// Compute inverse covariance
			gmm.InvCovs[j] = invertMatrix(flatMatrix(gmm.Covars[j]), dim)
		}

		// Check convergence
		invCovs, logDets = prepareGMM(gmm)
		newLogLik := gmmLogLikelihoodPrepared(gmm, data, invCovs, logDets)
		if math.Abs(newLogLik-oldLogLik) < tol {
			break
		}
	}

	return gmm
}

// gmmLogLikelihood returns log p(x | GMM) computed in the log domain, so
// densities that would underflow in the linear domain still contribute.
func gmmLogLikelihood(gmm *GMM, x []float64) float64 {
	invCovs, logDets := prepareGMM(gmm)
	logSum := math.Inf(-1)
	for j := 0; j < gmm.K; j++ {
		lik := math.Log(gmm.Weights[j]) +
			gaussianLogPDFPrepared(x, gmm.Means[j], invCovs[j], logDets[j])
		logSum = logAdd(logSum, lik)
	}
	return logSum
}

// prepareGMM computes each component's inverse covariance and log-determinant
// once per EM iteration; both are independent of the sample.
func prepareGMM(gmm *GMM) ([][]float64, []float64) {
	invCovs := make([][]float64, gmm.K)
	logDets := make([]float64, gmm.K)
	for j := 0; j < gmm.K; j++ {
		dim := gmm.Dim
		invCovs[j] = invertMatrix(flatMatrix(gmm.Covars[j]), dim)
		if invCovs[j] == nil {
			invCovs[j] = diagonalInverse(gmm.Covars[j], dim)
		}
		det := matrixDeterminant(gmm.Covars[j], dim)
		if det <= 0 {
			logDets[j] = math.Log(1e-300)
		} else {
			logDets[j] = math.Log(det)
		}
	}
	return invCovs, logDets
}

// gmmLogLikelihoodPrepared sums the log mixture likelihood over all samples
// using precomputed per-component inverses and log-determinants.
func gmmLogLikelihoodPrepared(gmm *GMM, data [][]float64, invCovs [][]float64, logDets []float64) float64 {
	total := 0.0
	for _, x := range data {
		logSum := math.Inf(-1)
		for j := 0; j < gmm.K; j++ {
			lik := math.Log(gmm.Weights[j]) +
				gaussianLogPDFPrepared(x, gmm.Means[j], invCovs[j], logDets[j])
			logSum = logAdd(logSum, lik)
		}
		total += logSum
	}
	return total
}

// gaussianLogPDFPrepared is the log density given the precomputed inverse
// covariance and log-determinant. Returns -Inf when the dims don't match.
func gaussianLogPDFPrepared(x, mean, invCov []float64, logDet float64) float64 {
	dim := len(x)
	if dim == 0 || len(mean) != dim || len(invCov) != dim*dim {
		return math.Inf(-1)
	}

	diff := make([]float64, dim)
	for i := 0; i < dim; i++ {
		diff[i] = x[i] - mean[i]
	}

	quadratic := 0.0
	for i := 0; i < dim; i++ {
		tmp := 0.0
		for j := 0; j < dim; j++ {
			tmp += invCov[i*dim+j] * diff[j]
		}
		quadratic += diff[i] * tmp
	}

	return -0.5*float64(dim)*math.Log(2*math.Pi) - 0.5*logDet - 0.5*quadratic
}

func diagonalInverse(covar [][]float64, dim int) []float64 {
	inv := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		if covar[i][i] > 1e-20 {
			inv[i*dim+i] = 1.0 / covar[i][i]
		} else {
			inv[i*dim+i] = 1e20
		}
	}
	return inv
}

func gaussianPDF(x, mean []float64, covar [][]float64) float64 {
	dim := len(x)
	if dim == 0 || len(mean) != dim {
		return 0
	}

	diff := make([]float64, dim)
	for i := 0; i < dim; i++ {
		diff[i] = x[i] - mean[i]
	}

	// Compute (x-mu)^T * Sigma^{-1} * (x-mu)
	det := matrixDeterminant(covar, dim)
	if det <= 0 {
		det = 1e-300
	}

	invCov := invertMatrix(flatMatrix(covar), dim)
	if invCov == nil {
		// Fallback to diagonal covariance
		invCov = diagonalInverse(covar, dim)
	}

	quadratic := 0.0
	for i := 0; i < dim; i++ {
		tmp := 0.0
		for j := 0; j < dim; j++ {
			tmp += invCov[i*dim+j] * diff[j]
		}
		quadratic += diff[i] * tmp
	}

	logNorm := -0.5 * float64(dim) * math.Log(2*math.Pi)
	logNorm -= 0.5 * math.Log(math.Abs(det))

	return math.Exp(logNorm - 0.5*quadratic)
}

func kmeansPPInit(data [][]float64, k int) [][]float64 {
	n := len(data)
	dim := len(data[0])
	centers := make([][]float64, k)

	// Choose first center randomly
	centers[0] = make([]float64, dim)
	copy(centers[0], data[rand.Intn(n)])

	for c := 1; c < k; c++ {
		// Compute distances to nearest center
		dists := make([]float64, n)
		totalDist := 0.0
		for i := 0; i < n; i++ {
			minDist := math.Inf(1)
			for j := 0; j < c; j++ {
				d := 0.0
				for d2 := 0; d2 < dim; d2++ {
					diff := data[i][d2] - centers[j][d2]
					d += diff * diff
				}
				if d < minDist {
					minDist = d
				}
			}
			dists[i] = minDist
			totalDist += minDist
		}

		// Weighted random selection
		r := rand.Float64() * totalDist
		cumsum := 0.0
		for i := 0; i < n; i++ {
			cumsum += dists[i]
			if cumsum >= r {
				centers[c] = make([]float64, dim)
				copy(centers[c], data[i])
				break
			}
		}
		if centers[c] == nil {
			centers[c] = make([]float64, dim)
			copy(centers[c], data[rand.Intn(n)])
		}
	}

	return centers
}

// Matrix utilities

func identityMatrix(dim int) [][]float64 {
	m := make([][]float64, dim)
	for i := range m {
		m[i] = make([]float64, dim)
		m[i][i] = 1.0
	}
	return m
}

func regularizeCovariance(cov [][]float64, dim int, epsilon float64) {
	for i := 0; i < dim; i++ {
		cov[i][i] += epsilon
	}
}

func flatMatrix(m [][]float64) []float64 {
	dim := len(m)
	flat := make([]float64, dim*dim)
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			flat[i*dim+j] = m[i][j]
		}
	}
	return flat
}

func matrixDeterminant(m [][]float64, n int) float64 {
	if n == 0 {
		return 1
	}
	if n == 1 {
		return m[0][0]
	}
	if n == 2 {
		return m[0][0]*m[1][1] - m[0][1]*m[1][0]
	}

	// Gaussian elimination (LU) — O(n^3). Laplace expansion would be O(n!)
	// and unusable beyond ~10 dimensions.
	a := make([][]float64, n)
	for i := range a {
		a[i] = append([]float64(nil), m[i]...)
	}

	det := 1.0
	for col := 0; col < n; col++ {
		// Partial pivoting
		pivot := col
		for row := col + 1; row < n; row++ {
			if math.Abs(a[row][col]) > math.Abs(a[pivot][col]) {
				pivot = row
			}
		}
		if math.Abs(a[pivot][col]) < 1e-300 {
			return 0
		}
		if pivot != col {
			a[col], a[pivot] = a[pivot], a[col]
			det = -det
		}
		det *= a[col][col]
		for row := col + 1; row < n; row++ {
			f := a[row][col] / a[col][col]
			for c := col + 1; c < n; c++ {
				a[row][c] -= f * a[col][c]
			}
		}
	}
	return det
}

func invertMatrix(flat []float64, n int) []float64 {
	if n == 0 {
		return nil
	}

	// Create augmented matrix [A | I]
	aug := make([][]float64, n)
	for i := 0; i < n; i++ {
		aug[i] = make([]float64, 2*n)
		for j := 0; j < n; j++ {
			aug[i][j] = flat[i*n+j]
		}
		aug[i][n+i] = 1.0
	}

	// Gauss-Jordan elimination
	for col := 0; col < n; col++ {
		// Find pivot
		maxRow := col
		maxVal := math.Abs(aug[col][col])
		for row := col + 1; row < n; row++ {
			if math.Abs(aug[row][col]) > maxVal {
				maxVal = math.Abs(aug[row][col])
				maxRow = row
			}
		}

		if maxVal < 1e-20 {
			return nil // Singular matrix
		}

		// Swap rows
		aug[col], aug[maxRow] = aug[maxRow], aug[col]

		// Scale pivot row
		pivot := aug[col][col]
		for j := 0; j < 2*n; j++ {
			aug[col][j] /= pivot
		}

		// Eliminate column
		for row := 0; row < n; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			for j := 0; j < 2*n; j++ {
				aug[row][j] -= factor * aug[col][j]
			}
		}
	}

	// Extract inverse
	inv := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inv[i*n+j] = aug[i][n+j]
		}
	}

	return inv
}

func logAdd(a, b float64) float64 {
	if math.IsInf(a, -1) {
		return b
	}
	if math.IsInf(b, -1) {
		return a
	}
	if a > b {
		return a + math.Log1p(math.Exp(b-a))
	}
	return b + math.Log1p(math.Exp(a-b))
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Sort scored results
type ScoredSpeaker struct {
	SpeakerID string
	Score     float64
}

func sortScored(results []ScoredSpeaker) {
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})
}

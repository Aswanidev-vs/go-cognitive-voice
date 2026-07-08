// Package ivector implements i-vector extraction and PLDA scoring for
// speaker verification. All implemented from scratch — no pre-trained models.
package ivector

import (
	"math"
	"math/rand"
)

// IVectorModel holds the trained i-vector extraction model.
type IVectorModel struct {
	UBM      *GMMModel       // Universal Background Model
	T        [][]float64     // Total variability matrix [dim][nComponents]
	mean     []float64       // UBM mean (for centering)
	nIVDim   int             // i-vector dimension
	dim      int             // feature dimension
}

// PLDAModel holds the PLDA scoring model.
type PLDAModel struct {
	Mu       []float64     // Global mean
	F        [][]float64   // Between-class factor [dim][nFactors]
	G        []float64     // Within-class covariance (diagonal)
	nFactors int
	dim      int
}

// GMMModel is a simplified GMM for the UBM.
type GMMModel struct {
	Weights []float64
	Means   [][]float64
	Covars  [][][]float64
	K       int
	Dim     int
}

// NewIVectorModel creates a new i-vector model.
func NewIVectorModel(nIVDim int) *IVectorModel {
	return &IVectorModel{nIVDim: nIVDim}
}

// FitUBM trains a GMM Universal Background Model using EM.
func (m *IVectorModel) FitUBM(data [][]float64, k int) {
	n := len(data)
	if n == 0 || k == 0 {
		return
	}
	m.dim = len(data[0])
	if k > n {
		k = n
	}

	m.UBM = &GMMModel{
		K:       k,
		Dim:     m.dim,
		Weights: make([]float64, k),
		Means:   make([][]float64, k),
		Covars:  make([][][]float64, k),
	}

	// Initialize with k-means++ style
	centers := kmeansPP(data, k)
	for i := 0; i < k; i++ {
		m.UBM.Means[i] = make([]float64, m.dim)
		copy(m.UBM.Means[i], centers[i])
		m.UBM.Covars[i] = identityMatrix(m.dim)
		m.UBM.Weights[i] = 1.0 / float64(k)
	}

	// EM iterations
	gamma := make([][]float64, n)
	for i := range gamma {
		gamma[i] = make([]float64, k)
	}

	for iter := 0; iter < 30; iter++ {
		// E-step
		for i := 0; i < n; i++ {
			total := 0.0
			for j := 0; j < k; j++ {
				gamma[i][j] = m.UBM.Weights[j] * gaussianPDF(data[i], m.UBM.Means[j], m.UBM.Covars[j])
				total += gamma[i][j]
			}
			if total > 0 {
				for j := 0; j < k; j++ {
					gamma[i][j] /= total
				}
			}
		}

		// M-step
		for j := 0; j < k; j++ {
			Nj := 0.0
			for i := 0; i < n; i++ {
				Nj += gamma[i][j]
			}
			if Nj < 1e-10 {
				continue
			}
			m.UBM.Weights[j] = Nj / float64(n)

			for d := 0; d < m.dim; d++ {
				sum := 0.0
				for i := 0; i < n; i++ {
					sum += gamma[i][j] * data[i][d]
				}
				m.UBM.Means[j][d] = sum / Nj
			}

			for d1 := 0; d1 < m.dim; d1++ {
				for d2 := 0; d2 < m.dim; d2++ {
					sum := 0.0
					for i := 0; i < n; i++ {
						v1 := data[i][d1] - m.UBM.Means[j][d1]
						v2 := data[i][d2] - m.UBM.Means[j][d2]
						sum += gamma[i][j] * v1 * v2
					}
					m.UBM.Covars[j][d1][d2] = sum / Nj
				}
			}
			// Regularize
			for d := 0; d < m.dim; d++ {
				m.UBM.Covars[j][d][d] += 1e-6
			}
		}
	}

	// Compute global mean
	m.mean = make([]float64, m.dim)
	for _, d := range data {
		for i := 0; i < m.dim; i++ {
			m.mean[i] += d[i]
		}
	}
	for i := range m.mean {
		m.mean[i] /= float64(n)
	}
}

// FitT trains the total variability matrix T using the EM algorithm.
func (m *IVectorModel) FitT(data [][]float64) {
	if m.UBM == nil {
		return
	}

	k := m.UBM.K
	n := m.nIVDim

	// Initialize T randomly
	m.T = make([][]float64, m.dim)
	for i := range m.T {
		m.T[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			m.T[i][j] = rand.NormFloat64() * 0.01
		}
	}

	// Simplified EM for T matrix
	for iter := 0; iter < 20; iter++ {
		for _, utterance := range data {
			// Compute sufficient statistics for this utterance
			// (simplified — use frame-level statistics)
			if len(utterance) < m.dim {
				continue
			}

			// Compute posterior for each GMM component
			posterior := make([]float64, k)
			total := 0.0
			for j := 0; j < k; j++ {
				posterior[j] = m.UBM.Weights[j] * gaussianPDF(utterance, m.UBM.Means[j], m.UBM.Covars[j])
				total += posterior[j]
			}
			if total > 0 {
				for j := 0; j < k; j++ {
					posterior[j] /= total
				}
			}

			// Compute i-vector for this utterance
			iv := m.extractIVector(utterance, posterior)

			// Update T (simplified gradient step)
			lr := 0.001
			for j := 0; j < k; j++ {
				if posterior[j] < 0.01 {
					continue
				}
				for d := 0; d < m.dim; d++ {
					diff := utterance[d] - m.UBM.Means[j][d]
					for f := 0; f < n; f++ {
						m.T[d][f] += lr * posterior[j] * diff * iv[f]
					}
				}
			}
		}
	}
}

// extractIVector extracts an i-vector from utterance features using EM.
func (m *IVectorModel) extractIVector(features []float64, posterior []float64) []float64 {
	n := m.nIVDim

	// Initialize i-vector
	iv := make([]float64, n)

	// Simplified posterior mean estimation
	for j := 0; j < m.UBM.K; j++ {
		if posterior[j] < 0.01 {
			continue
		}
		for d := 0; d < m.dim && d < len(features); d++ {
			diff := features[d] - m.UBM.Means[j][d]
			for f := 0; f < n; f++ {
				iv[f] += posterior[j] * m.T[d][f] * diff
			}
		}
	}

	return iv
}

// ExtractIVectorFromSequence extracts an i-vector from a sequence of features.
func (m *IVectorModel) ExtractIVectorFromSequence(features [][]float64) []float64 {
	if m.UBM == nil || len(features) == 0 {
		return make([]float64, m.nIVDim)
	}

	// Average posterior across frames
	avgPosterior := make([]float64, m.UBM.K)
	for _, f := range features {
		for j := 0; j < m.UBM.K; j++ {
			avgPosterior[j] += gaussianPDF(f, m.UBM.Means[j], m.UBM.Covars[j])
		}
	}
	total := 0.0
	for j := range avgPosterior {
		avgPosterior[j] *= m.UBM.Weights[j]
		total += avgPosterior[j]
	}
	if total > 0 {
		for j := range avgPosterior {
			avgPosterior[j] /= total
		}
	}

	// Use mean feature for extraction
	meanFeat := make([]float64, m.dim)
	for _, f := range features {
		for d := 0; d < m.dim && d < len(f); d++ {
			meanFeat[d] += f[d]
		}
	}
	for d := range meanFeat {
		meanFeat[d] /= float64(len(features))
	}

	return m.extractIVector(meanFeat, avgPosterior)
}

// --- PLDA ---

// NewPLDAModel creates a new PLDA model.
func NewPLDAModel(nFactors int) *PLDAModel {
	return &PLDAModel{nFactors: nFactors}
}

// Fit trains PLDA on i-vectors grouped by speaker.
func (p *PLDAModel) Fit(speakerIVs map[string][][]float64) {
	// Collect all i-vectors
	var allIV [][]float64
	for _, ivs := range speakerIVs {
		allIV = append(allIV, ivs...)
	}
	if len(allIV) == 0 {
		return
	}

	p.dim = len(allIV[0])

	// Compute global mean
	p.Mu = make([]float64, p.dim)
	for _, iv := range allIV {
		for i := 0; i < p.dim; i++ {
			p.Mu[i] += iv[i]
		}
	}
	for i := range p.Mu {
		p.Mu[i] /= float64(len(allIV))
	}

	// Center data
	centered := make([][]float64, len(allIV))
	for i, iv := range allIV {
		centered[i] = make([]float64, p.dim)
		for j := 0; j < p.dim; j++ {
			centered[i][j] = iv[j] - p.Mu[j]
		}
	}

	// Compute within-class and between-class scatter
	W := make([]float64, p.dim) // diagonal within-class
	B := make([][]float64, p.dim)
	for i := range B {
		B[i] = make([]float64, p.dim)
	}

	for _, ivs := range speakerIVs {
		// Speaker mean
		smean := make([]float64, p.dim)
		for _, iv := range ivs {
			for j := 0; j < p.dim; j++ {
				smean[j] += iv[j] - p.Mu[j]
			}
		}
		for j := range smean {
			smean[j] /= float64(len(ivs))
		}

		// Between-class: outer product of speaker mean
		for i := 0; i < p.dim; i++ {
			for j := 0; j < p.dim; j++ {
				B[i][j] += smean[i] * smean[j]
			}
		}

		// Within-class: deviation from speaker mean
		for _, iv := range ivs {
			for j := 0; j < p.dim; j++ {
				d := (iv[j] - p.Mu[j]) - smean[j]
				W[j] += d * d
			}
		}
	}

	// Normalize
	nSpeakers := float64(len(speakerIVs))
	nTotal := float64(len(allIV))
	for i := range W {
		W[i] /= nTotal
		for j := range B[i] {
			B[i][j] /= nSpeakers
		}
	}

	// Extract F (between-class factors) via eigendecomposition of B
	// Simplified: use top eigenvectors of B
	p.F = make([][]float64, p.dim)
	for i := range p.F {
		p.F[i] = make([]float64, p.nFactors)
	}

	// Power iteration to find top eigenvectors
	for f := 0; f < p.nFactors; f++ {
		v := make([]float64, p.dim)
		for i := range v {
			v[i] = rand.NormFloat64()
		}

		for iter := 0; iter < 50; iter++ {
			// Multiply B * v
			bv := matVecMul(B, v)
			// Orthogonalize against previous eigenvectors
			for prev := 0; prev < f; prev++ {
				dot := 0.0
				for i := 0; i < p.dim; i++ {
					dot += bv[i] * p.F[i][prev]
				}
				for i := 0; i < p.dim; i++ {
					bv[i] -= dot * p.F[i][prev]
				}
			}
			// Normalize
			norm := 0.0
			for _, x := range bv {
				norm += x * x
			}
			norm = math.Sqrt(norm)
			if norm > 1e-10 {
				for i := range bv {
					bv[i] /= norm
				}
			}
			v = bv
		}

		for i := 0; i < p.dim; i++ {
			p.F[i][f] = v[i]
		}
	}

	// Set G (within-class) as diagonal
	p.G = make([]float64, p.dim)
	for i := range W {
		p.G[i] = W[i]
		if p.G[i] < 1e-10 {
			p.G[i] = 1e-10
		}
	}
}

// Score computes PLDA log-likelihood ratio for two i-vectors.
func (p *PLDAModel) Score(iv1, iv2 []float64) float64 {
	if p.F == nil || len(iv1) != p.dim || len(iv2) != p.dim {
		return 0
	}

	// Center
	c1 := make([]float64, p.dim)
	c2 := make([]float64, p.dim)
	for i := 0; i < p.dim; i++ {
		c1[i] = iv1[i] - p.Mu[i]
		c2[i] = iv2[i] - p.Mu[i]
	}

	// Simplified PLDA score: cosine similarity of projected vectors
	// (full PLDA would use matrix inversions, this is an approximation)
	sameScore := 0.0
	diffScore := 0.0

	for f := 0; f < p.nFactors; f++ {
		proj1 := 0.0
		proj2 := 0.0
		for i := 0; i < p.dim; i++ {
			proj1 += p.F[i][f] * c1[i]
			proj2 += p.F[i][f] * c2[i]
		}
		sameScore += proj1 * proj2
	}

	// Normalize by within-class variance
	for i := 0; i < p.dim; i++ {
		diffScore += c1[i]*c1[i] + c2[i]*c2[i]
	}
	diffScore /= 2.0 * float64(p.dim)

	if diffScore < 1e-10 {
		return 0
	}

	return sameScore / diffScore
}

// --- Utility functions ---

func matVecMul(m [][]float64, v []float64) []float64 {
	n := len(m)
	result := make([]float64, n)
	for i := 0; i < n; i++ {
		sum := 0.0
		for j := 0; j < len(v) && j < len(m[i]); j++ {
			sum += m[i][j] * v[j]
		}
		result[i] = sum
	}
	return result
}

func identityMatrix(dim int) [][]float64 {
	m := make([][]float64, dim)
	for i := range m {
		m[i] = make([]float64, dim)
		m[i][i] = 1.0
	}
	return m
}

func gaussianPDF(x, mean []float64, covar [][]float64) float64 {
	dim := len(x)
	if dim == 0 || len(mean) != dim {
		return 0
	}

	det := 1.0
	for i := 0; i < dim; i++ {
		det *= covar[i][i]
	}
	if det <= 0 {
		det = 1e-300
	}

	quadratic := 0.0
	for i := 0; i < dim; i++ {
		d := x[i] - mean[i]
		quadratic += d * d / covar[i][i]
	}

	logNorm := -0.5 * float64(dim) * math.Log(2*math.Pi)
	logNorm -= 0.5 * math.Log(math.Abs(det))

	return math.Exp(logNorm - 0.5*quadratic)
}

func kmeansPP(data [][]float64, k int) [][]float64 {
	n := len(data)
	dim := len(data[0])
	centers := make([][]float64, k)

	centers[0] = make([]float64, dim)
	copy(centers[0], data[rand.Intn(n)])

	for c := 1; c < k; c++ {
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

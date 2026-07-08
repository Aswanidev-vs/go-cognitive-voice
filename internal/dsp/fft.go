package dsp

import (
	"math"
	"math/cmplx"
)

// FFT computes the 1D discrete Fourier transform.
// The input length MUST be a power of 2. Returns complex coefficients.
func FFT(x []float64) []complex128 {
	n := len(x)
	if n == 0 {
		return nil
	}
	if n == 1 {
		return []complex128{complex(x[0], 0)}
	}
	if n&(n-1) != 0 {
		panic("fft: input length must be a power of 2")
	}

	// Cooley-Tukey radix-2 DIT
	return fftRecursive(x)
}

func fftRecursive(x []float64) []complex128 {
	n := len(x)
	if n <= 2 {
		if n == 1 {
			return []complex128{complex(x[0], 0)}
		}
		// n == 2
		return []complex128{
			complex(x[0]+x[1], 0),
			complex(x[0]-x[1], 0),
		}
	}

	half := n / 2
	even := make([]float64, half)
	odd := make([]float64, half)
	for i := 0; i < half; i++ {
		even[i] = x[2*i]
		odd[i] = x[2*i+1]
	}

	evenFFT := fftRecursive(even)
	oddFFT := fftRecursive(odd)

	result := make([]complex128, n)
	twiddleBase := -2 * math.Pi / float64(n)

	for k := 0; k < half; k++ {
		angle := twiddleBase * float64(k)
		twiddle := cmplx.Rect(1, angle)
		t := oddFFT[k] * twiddle
		result[k] = evenFFT[k] + t
		result[k+half] = evenFFT[k] - t
	}

	return result
}

// NextPowerOf2 returns the smallest power of 2 >= n.
func NextPowerOf2(n int) int {
	if n <= 0 {
		return 1
	}
	p := 1
	for p < n {
		p <<= 1
	}
	return p
}

// PowerSpectrum computes the squared magnitude spectrum from real samples.
// Pads to next power of 2, applies FFT, returns [0:N/2] magnitudes squared.
func PowerSpectrum(samples []float64) []float64 {
	n := NextPowerOf2(len(samples))
	padded := make([]float64, n)
	copy(padded, samples)

	fftResult := FFT(padded)
	half := n/2 + 1
	psd := make([]float64, half)
	for i := 0; i < half; i++ {
		psd[i] = cmplx.Abs(fftResult[i])
		psd[i] *= psd[i] // squared magnitude
	}
	return psd
}

// DFT computes a single frequency bin of the DFT.
// Used for targeted frequency analysis without full FFT.
func DFT(x []float64, k int) complex128 {
	n := len(x)
	var sum complex128
	twiddleBase := -2 * math.Pi / float64(n)
	for t := 0; t < n; t++ {
		angle := twiddleBase * float64(k) * float64(t)
		sum += complex(x[t], 0) * cmplx.Rect(1, angle)
	}
	return sum
}

// IDFT computes the inverse discrete Fourier transform.
func IDFT(X []complex128) []float64 {
	n := len(X)
	if n == 0 {
		return nil
	}
	result := make([]float64, n)
	twiddleBase := 2 * math.Pi / float64(n)
	for t := 0; t < n; t++ {
		var sum complex128
		for k := 0; k < n; k++ {
			angle := twiddleBase * float64(k) * float64(t)
			sum += X[k] * cmplx.Rect(1, angle)
		}
		result[t] = real(sum) / float64(n)
	}
	return result
}

// LogEnergy computes log of energy (with floor to avoid log(0)).
func LogEnergy(power []float64) float64 {
	sum := 0.0
	for _, v := range power {
		sum += v
	}
	if sum < 1e-20 {
		return -20.0 // log10(1e-20)
	}
	return math.Log10(sum)
}

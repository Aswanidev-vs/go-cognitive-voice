// Package neural implements a small feedforward neural network trained from
// scratch on enrollment data for speaker embedding extraction. No pre-trained
// models, no external ML frameworks — just math and backpropagation.
package neural

import (
	"math"
	"math/rand"
)

// Network is a feedforward neural network for speaker embedding.
type Network struct {
	layers   []Layer
	dimIn    int
	dimEmbed int
}

// Layer holds weights and biases for one fully-connected layer.
type Layer struct {
	W      [][]float64 // [out][in]
	B      []float64   // [out]
	dW     [][]float64 // gradient accumulators
	dB     []float64
	act    Activation
	mW, vW [][]float64 // Adam moments
	mB, vB []float64
}

// Activation function type.
type Activation int

const (
	ActReLU Activation = iota
	ActLeakyReLU
	ActTanh
)

// New creates a network with architecture [dimIn, ...hidden, dimEmbed].
// Example: New(42, 128, 64) creates 42→128→64.
func New(dimIn int, hidden []int, dimEmbed int, act Activation) *Network {
	sizes := append([]int{dimIn}, hidden...)
	sizes = append(sizes, dimEmbed)

	n := &Network{dimIn: dimIn, dimEmbed: dimEmbed}
	for i := 0; i < len(sizes)-1; i++ {
		n.layers = append(n.layers, newLayer(sizes[i], sizes[i+1], act))
	}
	return n
}

func newLayer(in, out int, act Activation) Layer {
	// He initialization for ReLU-family
	scale := math.Sqrt(2.0 / float64(in))
	l := Layer{
		W:   make([][]float64, out),
		B:   make([]float64, out),
		dW:  make([][]float64, out),
		dB:  make([]float64, out),
		mW:  make([][]float64, out),
		vW:  make([][]float64, out),
		mB:  make([]float64, out),
		vB:  make([]float64, out),
		act: act,
	}
	for i := 0; i < out; i++ {
		l.W[i] = make([]float64, in)
		l.dW[i] = make([]float64, in)
		l.mW[i] = make([]float64, in)
		l.vW[i] = make([]float64, in)
		for j := 0; j < in; j++ {
			l.W[i][j] = rand.NormFloat64() * scale
		}
	}
	return l
}

// Forward runs the network. Returns the embedding (output of last layer).
// Also caches intermediate activations for backprop.
func (n *Network) Forward(x []float64) []float64 {
	a := x
	for i := range n.layers {
		a = n.layers[i].forward(a)
	}
	return a
}

// ForwardEmbed normalizes the output to unit length (L2).
func (n *Network) ForwardEmbed(x []float64) []float64 {
	out := n.Forward(x)
	return l2norm(out)
}

func (l *Layer) forward(x []float64) []float64 {
	out := make([]float64, len(l.W))
	for i := range l.W {
		sum := l.B[i]
		for j, w := range l.W[i] {
			sum += w * x[j]
		}
		out[i] = sum
	}
	// Apply activation (except last layer — linear output)
	if l.act == ActReLU {
		for i := range out {
			if out[i] < 0 {
				out[i] = 0
			}
		}
	} else if l.act == ActLeakyReLU {
		for i := range out {
			if out[i] < 0 {
				out[i] *= 0.01
			}
		}
	} else if l.act == ActTanh {
		for i := range out {
			out[i] = math.Tanh(out[i])
		}
	}
	return out
}

// Embed computes the speaker embedding for a sequence of features.
// Averages the network output across all frames.
func (n *Network) Embed(features [][]float64) []float64 {
	if len(features) == 0 {
		return make([]float64, n.dimEmbed)
	}

	sum := make([]float64, n.dimEmbed)
	for _, f := range features {
		emb := n.ForwardEmbed(f)
		for i := range emb {
			sum[i] += emb[i]
		}
	}

	nframes := float64(len(features))
	for i := range sum {
		sum[i] /= nframes
	}

	return l2norm(sum)
}

func l2norm(v []float64) []float64 {
	norm := 0.0
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm < 1e-10 {
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

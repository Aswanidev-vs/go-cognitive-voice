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
	W   [][]float64 // [out][in]
	B   []float64   // [out]
	mW  [][]float64 // momentum accumulators
	act Activation
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
		mW:  make([][]float64, out),
		act: act,
	}
	for i := 0; i < out; i++ {
		l.W[i] = make([]float64, in)
		l.mW[i] = make([]float64, in)
		for j := 0; j < in; j++ {
			l.W[i][j] = rand.NormFloat64() * scale
		}
	}
	return l
}

// Forward runs the network and returns the embedding (output of last layer).
func (n *Network) Forward(x []float64) []float64 {
	a := x
	for i := range n.layers {
		a = n.layers[i].forward(a)
	}
	return a
}

// forwardCached runs the network and additionally returns each layer's
// activations for backpropagation: cache[0] is x, cache[i+1] is the output
// of layer i.
func (n *Network) forwardCached(x []float64) ([]float64, [][]float64) {
	cache := make([][]float64, len(n.layers)+1)
	cache[0] = x
	a := x
	for i := range n.layers {
		a = n.layers[i].forward(a)
		cache[i+1] = a
	}
	return a, cache
}

// backprop propagates the gradient of the loss w.r.t. the network output
// back through the layers, updating weights with SGD + momentum.
// cache comes from forwardCached.
func (n *Network) backprop(cache [][]float64, grad []float64, lr float64) {
	for i := len(n.layers) - 1; i >= 0; i-- {
		l := &n.layers[i]
		in, out := cache[i], cache[i+1]

		// Gradient w.r.t. the pre-activation via the activation derivative,
		// which for ReLU-family and tanh can be derived from the output.
		pre := make([]float64, len(grad))
		for o := range grad {
			var d float64
			switch l.act {
			case ActReLU:
				if out[o] > 0 {
					d = 1
				}
			case ActLeakyReLU:
				if out[o] >= 0 {
					d = 1
				} else {
					d = 0.01
				}
			case ActTanh:
				d = 1 - out[o]*out[o]
			}
			pre[o] = grad[o] * d
		}

		gradIn := make([]float64, len(in))
		for o := range l.W {
			for j := range l.W[o] {
				w := l.W[o][j]
				gradIn[j] += w * pre[o]
				l.mW[o][j] = 0.9*l.mW[o][j] + 0.1*pre[o]*in[j]
				l.W[o][j] -= lr * l.mW[o][j]
			}
			l.B[o] -= lr * pre[o]
		}
		grad = gradIn
	}
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
			if j >= len(x) {
				break // tolerate inputs shorter than the configured dim
			}
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

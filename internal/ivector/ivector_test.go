package ivector

import (
	"math"
	"math/rand"
	"testing"
)

func TestIVectorModel_FitUBM(t *testing.T) {
	data := make([][]float64, 100)
	for i := range data {
		data[i] = make([]float64, 10)
		for j := range data[i] {
			if i < 50 {
				data[i][j] = math.Sin(float64(j)) + rand.Float64()*0.1
			} else {
				data[i][j] = math.Cos(float64(j)) + rand.Float64()*0.1
			}
		}
	}

	model := NewIVectorModel(32)
	model.FitUBM(data, 4)

	if model.UBM == nil {
		t.Fatal("UBM should not be nil")
	}
	if model.UBM.K != 4 {
		t.Errorf("UBM.K: got %d, want 4", model.UBM.K)
	}
}

func TestIVectorModel_ExtractIVector(t *testing.T) {
	data := make([][]float64, 50)
	for i := range data {
		data[i] = make([]float64, 10)
		for j := range data[i] {
			data[i][j] = math.Sin(float64(i+j)) * 0.5
		}
	}

	model := NewIVectorModel(16)
	model.FitUBM(data, 4)
	model.FitT(data)

	iv := model.ExtractIVectorFromSequence(data[:10])
	if len(iv) != 16 {
		t.Errorf("i-vector dim: got %d, want 16", len(iv))
	}
}

func TestPLDAModel_Fit(t *testing.T) {
	speakerIVs := map[string][][]float64{
		"alice": {{1, 2, 3}, {1.1, 2.1, 3.1}},
		"bob":   {{4, 5, 6}, {4.1, 5.1, 6.1}},
	}

	plda := NewPLDAModel(2)
	plda.Fit(speakerIVs)

	if plda.F == nil {
		t.Fatal("F should not be nil")
	}
}

func TestPLDAModel_Score(t *testing.T) {
	speakerIVs := map[string][][]float64{
		"alice": {{1, 2, 3}, {1.1, 2.1, 3.1}},
		"bob":   {{4, 5, 6}, {4.1, 5.1, 6.1}},
	}

	plda := NewPLDAModel(2)
	plda.Fit(speakerIVs)

	sameScore := plda.Score([]float64{1, 2, 3}, []float64{1.1, 2.1, 3.1})
	diffScore := plda.Score([]float64{1, 2, 3}, []float64{4, 5, 6})

	if sameScore <= diffScore {
		t.Errorf("same speaker score (%f) should be > different speaker (%f)", sameScore, diffScore)
	}
}

func TestGaussianPDF(t *testing.T) {
	x := []float64{0, 0}
	mean := []float64{0, 0}
	covar := [][]float64{{1, 0}, {0, 1}}

	pdf := gaussianPDF(x, mean, covar)
	expected := 1.0 / (2 * math.Pi)
	if math.Abs(pdf-expected) > 0.01 {
		t.Errorf("gaussianPDF: got %f, want %f", pdf, expected)
	}
}

func TestKmeansPP(t *testing.T) {
	data := make([][]float64, 20)
	for i := range data {
		data[i] = make([]float64, 5)
		if i < 10 {
			data[i][0] = 1.0
		} else {
			data[i][0] = 10.0
		}
		for j := 1; j < 5; j++ {
			data[i][j] = float64(j)
		}
	}

	centers := kmeansPP(data, 2)
	if len(centers) != 2 {
		t.Fatalf("expected 2 centers, got %d", len(centers))
	}
}

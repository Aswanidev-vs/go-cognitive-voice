package speaker

import (
	"encoding/json"
	"os"
)

// modelData is the serializable representation of a speaker model.
type modelData struct {
	ID          string      `json:"id"`
	Name        string      `json:"name"`
	Dim         int         `json:"dim"`
	NumSamples  int         `json:"num_samples"`
	TrainMethod string      `json:"train_method"`
	Centroid    []float64   `json:"centroid,omitempty"`
	Covariance  [][]float64 `json:"covariance,omitempty"`
	InvCovFlat  []float64   `json:"inv_cov,omitempty"`
	Features    [][]float64 `json:"features,omitempty"`
	GMM         *gmmData    `json:"gmm,omitempty"`
}

type gmmData struct {
	Weights []float64   `json:"weights"`
	Means   [][]float64 `json:"means"`
	Covars  [][][]float64 `json:"covars"`
	InvCovs [][]float64 `json:"inv_covs"`
	K       int         `json:"k"`
	Dim     int         `json:"dim"`
}

func saveModel(model *SpeakerModel, path string) error {
	data := modelData{
		ID:          model.ID,
		Name:        model.Name,
		Dim:         model.Dim,
		NumSamples:  model.NumSamples,
		TrainMethod: model.TrainMethod,
		Centroid:    model.Centroid,
		Covariance:  model.Covariance,
		InvCovFlat:  model.InvCovFlat,
		Features:    model.Features,
	}

	if model.GMM != nil {
		data.GMM = &gmmData{
			Weights: model.GMM.Weights,
			Means:   model.GMM.Means,
			Covars:  model.GMM.Covars,
			InvCovs: model.GMM.InvCovs,
			K:       model.GMM.K,
			Dim:     model.GMM.Dim,
		}
	}

	jsonBytes, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, jsonBytes, 0644)
}

func loadModel(path string) (*SpeakerModel, error) {
	jsonBytes, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var data modelData
	if err := json.Unmarshal(jsonBytes, &data); err != nil {
		return nil, err
	}

	model := &SpeakerModel{
		ID:          data.ID,
		Name:        data.Name,
		Dim:         data.Dim,
		NumSamples:  data.NumSamples,
		TrainMethod: data.TrainMethod,
		Centroid:    data.Centroid,
		Covariance:  data.Covariance,
		InvCovFlat:  data.InvCovFlat,
		Features:    data.Features,
	}

	if data.GMM != nil {
		model.GMM = &GMM{
			Weights: data.GMM.Weights,
			Means:   data.GMM.Means,
			Covars:  data.GMM.Covars,
			InvCovs: data.GMM.InvCovs,
			K:       data.GMM.K,
			Dim:     data.GMM.Dim,
		}
	}

	return model, nil
}

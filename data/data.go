package data

import (
	_ "log"
	"math"
)

type Data struct {
	Attributes     []float64
	Distances      []float64
	Classification int
}

func New(attr []float64, numClass int) *Data {
	return &Data{
		Attributes:     attr,
		Distances:      make([]float64, numClass),
		Classification: 0,
	}
}

func (data *Data) updateDistances(centroids []*Data) {
	for i, centroid := range centroids {
		sumOfSquares := 0.0
		for j := range centroid.Attributes {
			sumOfSquares += math.Pow(data.Attributes[j]-centroid.Attributes[j], 2)
		}
		distance := math.Sqrt(sumOfSquares)
		data.Distances[i] = distance
	}
}

func (data *Data) UpdateClassification(centroids []*Data) {
	data.updateDistances(centroids)
	minDistance := math.Inf(1)
	for i, distance := range data.Distances {
		if distance < minDistance {
			minDistance = distance
			data.Classification = i
		}
	}
}

func (data *Data) Waverage(item *Data, weigth float64) {
	for i := range data.Attributes {
		data.Attributes[i] = data.Attributes[i]*(1.0-weigth) + item.Attributes[i]*weigth
	}
}

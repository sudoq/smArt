package data

import (
	"testing"
	_ "log"
)

func TestNew(t *testing.T){
	attribs := []float64{0.1, 0.2, 0.5}
	d := New(attribs, 3)
	if d.Attributes[0] != 0.1 || d.Attributes[1] != 0.2 || d.Attributes[2] != 0.5 {
		t.Fail()
	}
}

func TestUpdateDistances(t *testing.T){
	d1 := New([]float64{0.1, 0.2, 0.5}, 2)
	c1 := New([]float64{0.0,0.0,0.0}, 2)
	c2 := New([]float64{1.0,1.0,1.0}, 2)
	centroids := []*Data{c1, c2}
	d1.updateDistances(centroids)
}

func TestUpdateClassification(t *testing.T){
	d0 := New([]float64{0.1, 0.2, 0.5}, 2)
	c0 := New([]float64{1.0,1.0,1.0}, 2)
	c1 := New([]float64{0.0,0.0,0.0}, 2)
	centroids := []*Data{c0, c1}
	d0.UpdateClassification(centroids)
	if d0.Classification != 1 {
		t.Fail()
	}
}

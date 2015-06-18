package main

import (
	"fmt"
	"math/rand"
	"time"
	"github.com/SudoQ/smArt/data"
)

func main(){
	rand.Seed(time.Now().UTC().UnixNano())

	K := 5
	dataSet := make([]*data.Data, 0)
	for i := 0; i < 10; i++ {
		d := data.New([]float64{rand.Float64(), rand.Float64(), rand.Float64()}, K)
		dataSet = append(dataSet, d)
	}

	centroids := make([]*data.Data, 0)
	for i:=0; i<K; i++ {
		c := data.New([]float64{rand.Float64(), rand.Float64(), rand.Float64()}, K)
		centroids = append(centroids, c)
	}

	for _, v := range(dataSet) {
		v.UpdateClassification(centroids)
	}

	for _, el := range(centroids){
		el.UpdateClassification(centroids)
		fmt.Println(el)
	}

	for _, el := range(dataSet){
		fmt.Println(el)
	}
}

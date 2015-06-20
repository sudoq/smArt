package main

import (
	"fmt"
	"math/rand"
	"github.com/SudoQ/smArt/data"
	"os"
	"flag"

	//"encoding/base64"
	"image"
	"log"
	//"strings"

	_ "image/gif"
	"image/png"
	_"image/jpeg"
	"time"
	"image/color"
	//"image/draw"
)

func loadImage(filename string) []*data.Data{
	// Decode the JPEG data. If reading from file, create a reader with

	reader, err := os.Open(filename)
	if err != nil {
	    log.Fatal(err)
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()

	dataSet := make([]*data.Data, 0)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := m.At(x, y).RGBA()
			// A color's RGBA method returns values in the range [0, 65535].
			// Shifting by 8 reduces this to the range [0, 255].
			a0 := float64(r>>8)
			a1 := float64(g>>8)
			a2 := float64(b>>8)
			d := data.New([]float64{a0, a1, a2}, 5)
			dataSet = append(dataSet, d)

		}
	}
	return dataSet
}

func saveCentroids(centroids []*data.Data, filename string){
	outfile, err := os.Create(filename)
	if err != nil {
		log.Println(err)
		return
	}
	imgRect := image.Rect(0, 0, 100, 100)
	img := image.NewRGBA(imgRect)
	for y := 0; y < 100; y += 1 {
		ci := int((float64(y)/100)*5);
		r := uint8(centroids[ci].Attributes[0])
		g := uint8(centroids[ci].Attributes[1])
		b := uint8(centroids[ci].Attributes[2])
		for x := 0; x < 100; x += 1 {
			img.SetRGBA(x,y,color.RGBA{r,g,b,255})
		}
	}

	err = png.Encode(outfile, img)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Generated image to %s\n",filename)
}

func applyModel(inputFilename, outputFilename string, centroids []*data.Data){
	reader, err := os.Open(inputFilename)
	if err != nil {
	    log.Fatal(err)
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()
	writer, err := os.Create(outputFilename)
	if err != nil {
		log.Println(err)
		return
	}
	defer writer.Close()

	imgRect := image.Rect(bounds.Min.X, bounds.Min.Y, bounds.Max.X, bounds.Max.Y)
	img := image.NewRGBA(imgRect)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := m.At(x, y).RGBA()
			a0 := float64(r>>8)
			a1 := float64(g>>8)
			a2 := float64(b>>8)
			d := data.New([]float64{a0, a1, a2}, 5)
			d.UpdateClassification(centroids)
			class := d.Classification
			b0 := uint8(centroids[class].Attributes[0])
			b1 := uint8(centroids[class].Attributes[1])
			b2 := uint8(centroids[class].Attributes[2])
			img.SetRGBA(x,y,color.RGBA{b0, b1, b2, 255})
		}
	}
	err = png.Encode(writer, img)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Generated image to %s\n",outputFilename)
}

var trainingFilename string
var evalFilename string
var paletteFilename string
var resultFilename string
func init() {
	flag.StringVar(&trainingFilename, "train", "default_input.png", "Input training filename")
	flag.StringVar(&evalFilename, "eval", "default_eval.png", "Input evaluation filename")
	flag.StringVar(&paletteFilename, "pal", "default_palette.png", "Output palette filename")
	flag.StringVar(&resultFilename, "result", "default_output.png", "Output filename")
}

func main(){
	flag.Parse()
	dataSet := loadImage(trainingFilename)
	fmt.Println(len(dataSet))
	rand.Seed(time.Now().UTC().UnixNano())

	K := 5
	centroids := []*data.Data{
		data.New([]float64{0.0, 0.0, 0.0}, K),  // Black
		data.New([]float64{255.0, 255.0, 255.0}, K),	// White
		data.New([]float64{255.0, 0.0, 0.0}, K),	// Red
		data.New([]float64{0.0, 255.0, 0.0}, K),	// Green
		data.New([]float64{0.0, 0.0, 255.0}, K),	// Blue
	}

	for n:=0; n<20; n++ {
		for _, v := range(dataSet) {
			v.UpdateClassification(centroids)
		}

		for _, v := range(centroids){
			v.UpdateClassification(centroids)
		}

		cCount := make(map[int]float64)
		for _, dataItem := range(dataSet) {
			ci := dataItem.Classification
			cCount[ci] += 1.0
			centroids[ci].Waverage(dataItem, 1.0/cCount[ci])
		}

	}
	for i, item := range(centroids) {
		r := uint32(item.Attributes[0])
		g := uint32(item.Attributes[1])
		b := uint32(item.Attributes[2])
		fmt.Printf("Color %d: (%d, %d, %d)\n", i,r,g,b)
	}
	saveCentroids(centroids, paletteFilename)
	applyModel(evalFilename,resultFilename,centroids)
}

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

	//reader := base64.NewDecoder(base64.StdEncoding, strings.NewReader(data))
	m, _, err := image.Decode(reader)
	if err != nil {
		log.Fatal(err)
	}
	bounds := m.Bounds()

	dataSet := make([]*data.Data, 0)

	// Calculate a 16-bin histogram for m's red, green, blue and alpha components.
	//
	// An image's bounds do not necessarily start at (0, 0), so the two loops start
	// at bounds.Min.Y and bounds.Min.X. Looping over Y first and X second is more
	// likely to result in better memory access patterns than X first and Y second.
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
	// Print the results.
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
	//draw.Draw(img, img.Bounds(), &image.Uniform{color.White}, image.ZP, draw.Src)
	for y := 0; y < 100; y += 1 {
		ci := int((float64(y)/100)*5);
		r := uint8(centroids[ci].Attributes[0])
		g := uint8(centroids[ci].Attributes[1])
		b := uint8(centroids[ci].Attributes[2])
		for x := 0; x < 100; x += 1 {
			img.SetRGBA(x,y,color.RGBA{r,g,b,255})
			//fill := &image.Uniform{color.RGBA{r, g, b, 255}}
			//draw.Draw(img, image.Rect(x, y, x+10, y+10), fill, image.ZP, draw.Src)
		}
	}

	err = png.Encode(outfile, img)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Generated image to %s\n",filename)
}

var inFilename string
var outFilename string
func init() {
	flag.StringVar(&inFilename, "infile", "default_in.png", "Input filename")
	flag.StringVar(&outFilename, "outfile", "default_out.png", "Output filename")
}

func main(){
	flag.Parse()
	dataSet := loadImage(inFilename)
	fmt.Println(len(dataSet))
	rand.Seed(time.Now().UTC().UnixNano())

	// Read image

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
	saveCentroids(centroids, outFilename)
}

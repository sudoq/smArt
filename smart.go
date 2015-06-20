package main

import (
	"flag"
	"fmt"
	"github.com/SudoQ/smArt/data"
	"math/rand"
	"math"
	"os"

	"image"
	"log"

	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"runtime"
	"sync"
	"time"
)

func loadTrainingImage(filename string) ([]*data.Data, int, int) {
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
			a0 := float64(r >> 8)
			a1 := float64(g >> 8)
			a2 := float64(b >> 8)
			dataItem := data.New([]float64{a0, a1, a2}, 5)
			dataSet = append(dataSet, dataItem)
		}
	}
	return dataSet, bounds.Dx(), bounds.Dy()
}

func saveCentroids(centroids []*data.Data, filename string) {
	outfile, err := os.Create(filename)
	if err != nil {
		log.Println(err)
		return
	}
	imgRect := image.Rect(0, 0, 100, 100)
	img := image.NewRGBA(imgRect)
	for y := 0; y < 100; y += 1 {
		ci := int((float64(y) / 100) * 5)
		r := uint8(centroids[ci].Attributes[0])
		g := uint8(centroids[ci].Attributes[1])
		b := uint8(centroids[ci].Attributes[2])
		for x := 0; x < 100; x += 1 {
			img.SetRGBA(x, y, color.RGBA{r, g, b, 255})
		}
	}

	err = png.Encode(outfile, img)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Generated image to %s\n", filename)
}

func applyModel(inputFilename, outputFilename string, centroids []*data.Data) {
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
			r0 := float64(r >> 8)
			g0 := float64(g >> 8)
			b0 := float64(b >> 8)
			dataItem := data.New([]float64{r0, g0, b0}, len(centroids))
			dataItem.UpdateClassification(centroids)
			class := dataItem.Classification
			r1 := uint8(centroids[class].Attributes[0])
			g1 := uint8(centroids[class].Attributes[1])
			b1 := uint8(centroids[class].Attributes[2])
			img.SetRGBA(x, y, color.RGBA{r1, g1, b1, 255})
		}
	}
	err = png.Encode(writer, img)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}

	fmt.Printf("Generated image to %s\n", outputFilename)
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
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	flag.Parse()
	dataSet, width, height := loadTrainingImage(trainingFilename)
	fmt.Println(len(dataSet))
	rand.Seed(time.Now().UTC().UnixNano())

	// TODO Read initial centroids from file
	K := 5
	centroids := []*data.Data{
		data.New([]float64{0.0, 0.0, 0.0}, K),       // Black
		data.New([]float64{255.0, 255.0, 255.0}, K), // White
		data.New([]float64{255.0, 0.0, 0.0}, K),     // Red
		data.New([]float64{0.0, 255.0, 0.0}, K),     // Green
		data.New([]float64{0.0, 0.0, 255.0}, K),     // Blue
	}
	fmt.Printf("%d * %d = %d\n", width, height, width*height)
	wg := sync.WaitGroup{}
	sections := 8
	centroidsChanged := true
	var n int
	for n = 0; n < 30 && centroidsChanged; n++ {
		// TODO Check if the centroids is unchanged, if so, stop
		sublength := height / sections // 640 / 8 = 80
		wg.Add(sections)
		for s := 0; s < sections; s++ {
			dh := s * sublength
			go func(dh int) {
				for h := dh; h < dh+(sublength); h++ {
					for w := 0; w < width; w++ {
						dataSet[(h*width)+w].UpdateClassification(centroids)
					}
				}
			}(dh)
			wg.Done()
		}
		wg.Wait()
		/*
			wg.Add(height)
			for h:=0; h<height; h++ {
				go func(h int) {
					for w:=0; w<width; w++ {
						dataSet[(h*width)+w].UpdateClassification(centroids)
					}
					wg.Done()
				}(h)
			}
			wg.Wait()
		*/

		for _, v := range dataSet {
			v.UpdateClassification(centroids)
		}

		currentAttributes := make([][]float64,0)
		for _, c := range(centroids){
			a0 := c.Attributes[0]
			a1 := c.Attributes[1]
			a2 := c.Attributes[2]
			currentAttributes = append(currentAttributes, []float64{a0, a1, a2})
		}

		for _, v := range centroids {
			v.UpdateClassification(centroids)
		}

		cCount := make(map[int]float64)
		for _, dataItem := range dataSet {
			ci := dataItem.Classification
			cCount[ci] += 1.0
			centroids[ci].Waverage(dataItem, 1.0/cCount[ci])
		}

		centroidsChanged = false
		for i, c := range(centroids){
			for j := 0; j<3; j++ {
				if math.Abs(currentAttributes[i][j] - c.Attributes[j]) > 0.5 {
					centroidsChanged = true
				}
			}
		}

	}

	fmt.Printf("Number of iterations: %d\n", n)
	for i, item := range centroids {
		r := uint32(item.Attributes[0])
		g := uint32(item.Attributes[1])
		b := uint32(item.Attributes[2])
		fmt.Printf("Color %d: (%d, %d, %d)\n", i, r, g, b)
	}
	saveCentroids(centroids, paletteFilename)
	applyModel(evalFilename, resultFilename, centroids)
}

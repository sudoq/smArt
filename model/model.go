package model

import (
	"encoding/csv"
	"fmt"
	"github.com/SudoQ/smArt/data"
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"math"
	"os"
	"strconv"
	"sync"
)

type Model struct {
	Centroids []*data.Data
}

func New() *Model {
	return &Model{
		Centroids: make([]*data.Data, 0),
	}
}

func (model *Model) Load(filename string) error {
	csvfile, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer csvfile.Close()
	reader := csv.NewReader(csvfile)
	rawCSVdata, err := reader.ReadAll()
	if err != nil {
		return err
	}
	k := len(rawCSVdata)
	centroids := make([]*data.Data, 0)
	for _, v := range rawCSVdata {
		a0, _ := strconv.ParseFloat(v[0], 64)
		a1, _ := strconv.ParseFloat(v[1], 64)
		a2, _ := strconv.ParseFloat(v[2], 64)
		centroids = append(centroids, data.New([]float64{a0, a1, a2}, k))
	}
	model.Centroids = centroids
	return nil
}

func (model *Model) Save(filename string) error {
	csvfile, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer csvfile.Close()
	writer := csv.NewWriter(csvfile)
	lines := make([][]string, 0)
	for _, c := range model.Centroids {
		line := make([]string, 3)
		for j, a := range c.Attributes {
			line[j] = fmt.Sprintf("%d.0", uint8(a))
		}
		lines = append(lines, line)
	}
	err = writer.WriteAll(lines)
	if err != nil {
		return err
	}
	return nil
}

func (model *Model) SaveCentroidsImage(filename string) error {
	outfile, err := os.Create(filename)
	if err != nil {
		return err
	}
	imgRect := image.Rect(0, 0, 100, 100)
	img := image.NewRGBA(imgRect)
	for y := 0; y < 100; y += 1 {
		ci := int((float64(y) / 100) * float64(len(model.Centroids)))
		r := uint8(model.Centroids[ci].Attributes[0])
		g := uint8(model.Centroids[ci].Attributes[1])
		b := uint8(model.Centroids[ci].Attributes[2])
		for x := 0; x < 100; x += 1 {
			img.SetRGBA(x, y, color.RGBA{r, g, b, 255})
		}
	}

	err = png.Encode(outfile, img)
	if err != nil {
		return err
	}

	return nil
}

func loadTrainingImage(filename string, numClasses int) ([]*data.Data, int, int, error) {
	// Open and read image
	reader, err := os.Open(filename)
	if err != nil {
		return nil, 0, 0, err
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	if err != nil {
		return nil, 0, 0, err
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
			// Store rgb values in data set
			dataItem := data.New([]float64{a0, a1, a2}, numClasses)
			dataSet = append(dataSet, dataItem)
		}
	}
	// Return data set, image width and height
	return dataSet, bounds.Dx(), bounds.Dy(), nil
}

func (model *Model) Train(filename string) error {
	dataSet, width, height, err := loadTrainingImage(filename, len(model.Centroids))
	if err != nil {
		return err
	}
	wg := sync.WaitGroup{}
	sections := 16
	centroidsChanged := true
	var n int
	for n = 0; n < 30 && centroidsChanged; n++ {
		fmt.Printf("Iteration %d...\n", n)
		sublength := width / sections // 640 / 8 = 80
		wg.Add(sections)
		for s := 0; s < sections; s++ {
			dh := s * sublength
			go func(dh int) {
				for h := dh; h < dh+(sublength); h++ {
					for w := 0; w < height; w++ {
						dataSet[(h*height)+w].UpdateClassification(model.Centroids)
					}
				}
				fmt.Printf("#")
				wg.Done()
			}(dh)
		}
		wg.Wait()

		currentAttributes := make([][]float64, 0)
		for _, c := range model.Centroids {
			a0 := c.Attributes[0]
			a1 := c.Attributes[1]
			a2 := c.Attributes[2]
			currentAttributes = append(currentAttributes, []float64{a0, a1, a2})
		}

		for _, v := range model.Centroids {
			v.UpdateClassification(model.Centroids)
		}

		cCount := make(map[int]float64)
		for i, dataItem := range dataSet {
			ci := dataItem.Classification
			cCount[ci] += 1.0
			model.Centroids[ci].Waverage(dataItem, 1.0/cCount[ci])
			if (i % len(dataSet) / 8) == 0 {
				fmt.Printf("#")
			}
		}
		fmt.Println()

		centroidsChanged = false
		for i, c := range model.Centroids {
			for j := 0; j < 3; j++ {
				if math.Abs(currentAttributes[i][j]-c.Attributes[j]) > 0.5 {
					centroidsChanged = true
				}
			}
		}
	}

	fmt.Printf("Number of iterations: %d\n", n)
	for i, item := range model.Centroids {
		r := uint32(item.Attributes[0])
		g := uint32(item.Attributes[1])
		b := uint32(item.Attributes[2])
		fmt.Printf("Color %d: (%d, %d, %d)\n", i, r, g, b)
	}
	return nil
}

func (model *Model) Classify(inputFilename, outputFilename string) error {
	reader, err := os.Open(inputFilename)
	if err != nil {
		return err
	}
	defer reader.Close()

	m, _, err := image.Decode(reader)
	if err != nil {
		return err
	}
	bounds := m.Bounds()
	writer, err := os.Create(outputFilename)
	if err != nil {
		return err
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
			dataItem := data.New([]float64{r0, g0, b0}, len(model.Centroids))
			dataItem.UpdateClassification(model.Centroids)
			class := dataItem.Classification
			r1 := uint8(model.Centroids[class].Attributes[0])
			g1 := uint8(model.Centroids[class].Attributes[1])
			b1 := uint8(model.Centroids[class].Attributes[2])
			img.SetRGBA(x, y, color.RGBA{r1, g1, b1, 255})
		}
	}
	err = png.Encode(writer, img)
	if err != nil {
		return err
	}
	return nil
}

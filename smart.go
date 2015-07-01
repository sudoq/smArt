package main

import (
	"flag"
	"fmt"
	"github.com/SudoQ/smArt/model"
	"log"
	"os"
	"path/filepath"
	"runtime"
)

var trainingTarget string
var classifyTarget string
var numClasses int
var maxIterations int

func init() {
	flag.StringVar(&trainingTarget, "t", "resources/default.png", "Input training image")
	flag.StringVar(&classifyTarget, "c", "resources/default.png", "Input classify image")
	flag.IntVar(&maxIterations, "n", 30, "Max number of iterations")
	flag.IntVar(&numClasses, "m", 5, "Number of classes")
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func errorGate(err error) {
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}
}

func splitExt(path string) (string, string) {
	ext := filepath.Ext(path)
	base := path
	name := base[:(len(base) - len(ext))]
	return name, ext
}

func main() {
	flag.Parse()

	trainingBase, _ := splitExt(trainingTarget)
	classifyBase, _ := splitExt(classifyTarget)

	outCSVfilename := trainingBase + ".csv"
	trainingFilename := trainingTarget
	evalFilename := classifyTarget
	paletteFilename := trainingBase + "_palette.png"
	resultFilename := classifyBase + "_out.png"

	m := model.New()
	errorGate(m.Train(trainingFilename, numClasses, maxIterations))
	fmt.Printf("Trained model from image %s\n", trainingFilename)
	errorGate(m.SaveCentroidsImage(paletteFilename))
	fmt.Printf("Saved model color palette to %s\n", paletteFilename)
	errorGate(m.Classify(evalFilename, resultFilename))
	fmt.Printf("Classified image %s and generated image %s\n", evalFilename, resultFilename)
	errorGate(m.Save(outCSVfilename))
	fmt.Printf("Saved model centroids to %s\n", outCSVfilename)
}

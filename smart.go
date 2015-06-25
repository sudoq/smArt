package main

import (
	"flag"
	"log"
	"fmt"
	"os"
	"runtime"
	"github.com/SudoQ/smArt/model"
)

var inCSVfilename string
var outCSVfilename string
var trainingFilename string
var evalFilename string
var paletteFilename string
var resultFilename string

func init() {
	flag.StringVar(&inCSVfilename, "incsv", "resources/centroids_in.csv", "Input centroids CSV filename")
	flag.StringVar(&outCSVfilename, "outcsv", "resources/centroids_out.csv", "Output centroids CSV filename")
	flag.StringVar(&trainingFilename, "train", "resources/default_input.png", "Input training filename")
	flag.StringVar(&evalFilename, "eval", "resources/default_eval.png", "Input evaluation filename")
	flag.StringVar(&paletteFilename, "pal", "resources/default_palette.png", "Output palette filename")
	flag.StringVar(&resultFilename, "result", "resources/default_output.png", "Output result filename")
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func errorGate(err error){
	if err != nil {
		log.Println(err)
		os.Exit(1)
	}
}

func main() {
	flag.Parse()
	m := model.New()
	errorGate(m.Load(inCSVfilename))
	fmt.Printf("Loaded model centroids from %s\n",inCSVfilename)
	errorGate(m.Train(trainingFilename))
	fmt.Printf("Trained model from image %s\n",trainingFilename)
	errorGate(m.SaveCentroidsImage(paletteFilename))
	fmt.Printf("Saved model color palette to %s\n", paletteFilename)
	errorGate(m.Classify(evalFilename, resultFilename))
	fmt.Printf("Classified image %s and generated image %s\n", evalFilename, resultFilename)
	errorGate(m.Save(outCSVfilename))
	fmt.Printf("Saved model centroids to %s\n",outCSVfilename)
}


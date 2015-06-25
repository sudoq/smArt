package main

import (
	"flag"
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
	flag.StringVar(&inCSVfilename, "incsv", "centroids_in.csv", "Input centroids CSV filename")
	flag.StringVar(&outCSVfilename, "outcsv", "centroids_out.csv", "Output centroids CSV filename")
	flag.StringVar(&trainingFilename, "train", "default_input.png", "Input training filename")
	flag.StringVar(&evalFilename, "eval", "default_eval.png", "Input evaluation filename")
	flag.StringVar(&paletteFilename, "pal", "default_palette.png", "Output palette filename")
	flag.StringVar(&resultFilename, "result", "default_output.png", "Output filename")
	runtime.GOMAXPROCS(runtime.NumCPU())
}
func main() {
	flag.Parse()
	m := model.New()
	m.Load(inCSVfilename)
	m.Train(trainingFilename)
	m.SaveCentroidsImage(paletteFilename)
	m.Classify(evalFilename, resultFilename)
	m.Save(outCSVfilename)
}


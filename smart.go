package main

import (
	/*
		"encoding/csv"
		"flag"
		"fmt"
		"image"
		"image/color"
		"image/png"
		_ "image/gif"
		_ "image/jpeg"
		"log"
		"math/rand"
		"math"
		"os"
		"strconv"
		"sync"
		"time"

		"github.com/SudoQ/smArt/data"
	*/
	"fmt"
	"flag"
	"runtime"
	"github.com/SudoQ/smArt/model"
)

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
	m := model.New()
	m.Load("centroids.csv")
	m.Train(trainingFilename)
	m.SaveCentroidsImage(paletteFilename)
	//m.Classify(evalFilename)
	m.Save("centroids_out.csv")
	fmt.Println(m)
}


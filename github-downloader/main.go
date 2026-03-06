package main

import (
	"fmt"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: github-downloader <github-url> [output-directory]")
		fmt.Println("Example: github-downloader https://github.com/mloopa/eolabs/tree/main/lab_1/data")
		fmt.Println("Example: github-downloader https://github.com/mloopa/eolabs/tree/main/lab_1/data ./downloads")
		os.Exit(1)
	}
	
	repoURL := os.Args[1]
	outputDir := "./downloads"
	
	if len(os.Args) >= 3 {
		outputDir = os.Args[2]
	}
	
	downloader := NewGitHubDownloader()
	
	if err := downloader.DownloadFolder(repoURL, outputDir); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
}

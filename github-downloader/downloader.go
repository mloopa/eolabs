package main

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/google/go-github/v56/github"
)

type GitHubDownloader struct {
	client *github.Client
	ctx    context.Context
}

func NewGitHubDownloader() *GitHubDownloader {
	return &GitHubDownloader{
		client: github.NewClient(nil),
		ctx:    context.Background(),
	}
}

func (gd *GitHubDownloader) parseGitHubURL(repoURL string) (owner, repo, path string, err error) {
	// Parse GitHub URL format: https://github.com/owner/repo/tree/branch/path/to/folder
	pattern := `^https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+)$`
	re := regexp.MustCompile(pattern)
	matches := re.FindStringSubmatch(repoURL)

	if len(matches) != 5 {
		return "", "", "", fmt.Errorf("invalid GitHub URL format. Expected: https://github.com/owner/repo/tree/branch/path")
	}

	return matches[1], matches[2], matches[4], nil
}

func (gd *GitHubDownloader) getRepositoryContents(owner, repo, path string) ([]*github.RepositoryContent, error) {
	_, directoryContent, _, err := gd.client.Repositories.GetContents(gd.ctx, owner, repo, path, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to get repository contents: %w", err)
	}

	var files []*github.RepositoryContent
	for _, item := range directoryContent {
		if *item.Type == "file" {
			files = append(files, item)
		} else if *item.Type == "dir" {
			// Recursively get files from subdirectories
			subFiles, err := gd.getRepositoryContents(owner, repo, *item.Path)
			if err != nil {
				return nil, err
			}
			files = append(files, subFiles...)
		}
	}

	return files, nil
}

type LFSPointer struct {
	Version string `json:"version"`
	OID     string `json:"oid"`
	Size    int64  `json:"size"`
}

func (gd *GitHubDownloader) downloadFile(rawURL, localPath, owner, repo, filePath string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(localPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	// Download the file
	resp, err := http.Get(rawURL)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: status code %d", resp.StatusCode)
	}

	// Read the content to check if it's an LFS pointer
	content, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("failed to read file content: %w", err)
	}

	// Check if this is a Git LFS pointer file
	contentStr := string(content)
	if strings.HasPrefix(contentStr, "version https://git-lfs.github.com/spec/") {
		// Parse LFS pointer
		lines := strings.Split(contentStr, "\n")
		var lfsPointer LFSPointer
		for _, line := range lines {
			if strings.HasPrefix(line, "oid sha256:") {
				lfsPointer.OID = strings.TrimPrefix(line, "oid sha256:")
			} else if strings.HasPrefix(line, "size ") {
				_, err := fmt.Sscanf(line, "size %d", &lfsPointer.Size)
				if err != nil {
					return fmt.Errorf("failed to parse LFS size: %w", err)
				}
			}
		}

		if lfsPointer.OID == "" {
			return fmt.Errorf("invalid LFS pointer: missing OID")
		}

		// Download the actual LFS file using GitHub's media URL
		mediaURL := fmt.Sprintf("https://media.githubusercontent.com/media/%s/%s/main/%s", owner, repo, filePath)
		return gd.downloadDirectFile(mediaURL, localPath)
	}

	// Regular file, write content directly
	file, err := os.Create(localPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", localPath, err)
	}
	defer file.Close()

	_, err = file.Write(content)
	if err != nil {
		return fmt.Errorf("failed to write file content: %w", err)
	}

	return nil
}

func (gd *GitHubDownloader) downloadDirectFile(url, localPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download file: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("failed to download file: status code %d", resp.StatusCode)
	}

	file, err := os.Create(localPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", localPath, err)
	}
	defer file.Close()

	_, err = io.Copy(file, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file content: %w", err)
	}

	return nil
}

func (gd *GitHubDownloader) DownloadFolder(repoURL, outputDir string) error {
	owner, repo, path, err := gd.parseGitHubURL(repoURL)
	if err != nil {
		return err
	}

	fmt.Printf("Fetching files from %s/%s/%s...\n", owner, repo, path)

	// Get all files in the directory
	files, err := gd.getRepositoryContents(owner, repo, path)
	if err != nil {
		return err
	}

	if len(files) == 0 {
		fmt.Println("No files found in the specified directory.")
		return nil
	}

	fmt.Printf("Found %d files. Starting download...\n", len(files))

	// Download each file
	for i, file := range files {
		// Construct raw URL
		rawURL := fmt.Sprintf("https://raw.githubusercontent.com/%s/%s/main/%s", owner, repo, *file.Path)

		// Construct local path
		relativePath := strings.TrimPrefix(*file.Path, path)
		relativePath = strings.TrimPrefix(relativePath, "/")
		localPath := filepath.Join(outputDir, relativePath)

		fmt.Printf("Downloading %d/%d: %s\n", i+1, len(files), *file.Name)

		if err := gd.downloadFile(rawURL, localPath, owner, repo, *file.Path); err != nil {
			fmt.Printf("Error downloading %s: %v\n", *file.Name, err)
			continue
		}

		fmt.Printf("✓ Downloaded %s\n", *file.Name)
	}

	fmt.Printf("Download completed! Files saved to: %s\n", outputDir)
	return nil
}

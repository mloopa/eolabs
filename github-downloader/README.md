# GitHub Downloader

A Go CLI tool that downloads all files from a GitHub repository folder using raw file URLs.

## Features

- Downloads all files from a GitHub repository directory
- Recursively handles subdirectories
- Preserves directory structure
- Shows download progress
- Error handling for failed downloads

## Installation

1. Clone or download this project
2. Install dependencies:
   ```bash
   go mod tidy
   ```
3. Build the executable:
   ```bash
   go build -o github-downloader
   ```

## Usage

### Basic Usage
```bash
./github-downloader <github-url>
```

### With Custom Output Directory
```bash
./github-downloader <github-url> <output-directory>
```

### Examples

Download files to default `./downloads` directory:
```bash
./github-downloader https://github.com/mloopa/eolabs/tree/main/lab_1/data
```

Download files to custom directory:
```bash
./github-downloader https://github.com/mloopa/eolabs/tree/main/lab_1/data ./my-downloads
```

## URL Format

The tool expects GitHub URLs in the following format:
```
https://github.com/owner/repo/tree/branch/path/to/folder
```

Examples:
- `https://github.com/mloopa/eolabs/tree/main/lab_1/data`
- `https://github.com/google/go-github/tree/main/github`
- `https://github.com/golang/go/tree/master/src`

## How It Works

1. Parses the GitHub URL to extract owner, repository, and path
2. Uses GitHub API to list all files in the specified directory
3. Recursively fetches files from subdirectories
4. Downloads each file using raw GitHub URLs
5. Preserves the original directory structure locally

## Dependencies

- `github.com/google/go-github/v56` - GitHub API client
- `golang.org/x/oauth2` - OAuth2 support (for potential future authentication)

## Error Handling

The tool will:
- Skip files that fail to download and continue with others
- Show error messages for failed downloads
- Create directories as needed
- Validate URL format before processing

## Limitations

- Works with public repositories only
- Requires exact GitHub URL format with branch/tree segment
- Downloads files from the `main` branch by default in raw URLs

param (
    [string]$commitMessage
)

if (-not $commitMessage) {
    Write-Host "Usage: .\run_versioning.ps1 <python_file> <commit_message>"
    exit 1
}

try {
    # Run the specified Python script with the commit message
    python version_new_data.py $commitMessage
} catch {
    Write-Host "An error occurred: $($_.Exception.Message)"
    exit 1
}

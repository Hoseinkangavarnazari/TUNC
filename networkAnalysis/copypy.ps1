#  .\networkAnalysis\copypy.ps1 .\networkAnalysis\pr_new.py 5

# Check if the correct number of arguments is provided
if ($args.Count -ne 2) {
    Write-Host "Usage: $PSCommandPath <filename.py> <number of copies>"
    exit
}

# Assign arguments to variables
$filename = $args[0]
$copies = $args[1]

# Extract the directory path and the base name of the file
$dirPath = Split-Path -Parent $filename
$baseName = [System.IO.Path]::GetFileNameWithoutExtension($filename)

# Loop to create copies in the same directory as the original file
1..$copies | ForEach-Object {
    $newFileName = Join-Path $dirPath ("temp$_" + "_" + $baseName + ".py")
    Copy-Item $filename $newFileName
}

param(
    [Parameter(Mandatory = $true)]
    [string]$InputDocx,
    [Parameter(Mandatory = $true)]
    [string]$OutputPdf,
    [Parameter(Mandatory = $true)]
    [string]$StatusFile
)

$ErrorActionPreference = 'Stop'
$word = $null
$doc = $null

try {
    Set-Content -LiteralPath $StatusFile -Value "STARTED" -Encoding UTF8
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.Options.SaveNormalPrompt = $false
    Set-Content -LiteralPath $StatusFile -Value "WORD_READY" -Encoding UTF8
    $doc = $word.Documents.Open($InputDocx, $false, $true, $false)
    Set-Content -LiteralPath $StatusFile -Value "DOCUMENT_OPEN" -Encoding UTF8
    $doc.ExportAsFixedFormat($OutputPdf, 17)
    Set-Content -LiteralPath $StatusFile -Value "PDF_EXPORTED" -Encoding UTF8
    $doc.Close($false)
    $doc = $null
    $word.Quit()
    $word = $null
    Set-Content -LiteralPath $StatusFile -Value "DONE" -Encoding UTF8
}
catch {
    $message = $_.Exception.ToString()
    Set-Content -LiteralPath $StatusFile -Value ("ERROR`n" + $message) -Encoding UTF8
    if ($doc -ne $null) {
        try { $doc.Close($false) } catch {}
    }
    if ($word -ne $null) {
        try { $word.Quit() } catch {}
    }
    exit 1
}

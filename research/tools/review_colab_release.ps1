param(
    [ValidateSet('Review','Stage','Commit','Package')][string]$Action = 'Review',
    [string]$ApprovedHead,
    [string]$Output
)
# SENIOR-ONLY helper. Default is review. No push or Windows pin/lock changes.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$pythonPath = Join-Path (Split-Path $repoRoot -Parent) '.venvs\deepaa_data2026_repro\Scripts\python.exe'
$migrationFiles = @(
    'research/code/colab_transfer.py', 'research/code/colab_runtime.py',
    'research/tools/build_colab_transfer.py', 'research/tools/colab_migration.py',
    'research/tools/colab_gpu_worker.py', 'research/tools/review_colab_release.ps1',
    'research/tests/test_colab_migration.py', 'research/notebooks/stability_v1_colab.ipynb',
    'research/requirements_colab.txt', 'research/COLAB_MIGRATION.md',
    'docs/decisions/decision_log.md', 'docs/project_state.md',
    'docs/experiments/experiment_registry.md'
)
Push-Location -LiteralPath $repoRoot
try {
    if ($Action -eq 'Review') {
        git --no-optional-locks status --short
        if ($LASTEXITCODE -ne 0) { throw 'git status failed' }
        git --no-optional-locks diff --check
        if ($LASTEXITCODE -ne 0) { throw 'diff whitespace check failed' }
        git --no-optional-locks diff -- docs/decisions/decision_log.md docs/project_state.md docs/experiments/experiment_registry.md
        if ($LASTEXITCODE -ne 0) { throw 'git diff failed' }
        foreach ($migrationFile in $migrationFiles) {
            Get-Item -LiteralPath $migrationFile | Select-Object FullName, Length
        }
        & $pythonPath -B -m pytest research/tests/test_colab_migration.py research/tests/test_workflow.py research/tests/test_output_contract.py research/tests/test_degenerate_recovery.py -q -p no:cacheprovider
        if ($LASTEXITCODE -ne 0) { throw 'Focused tests failed' }
        Write-Output 'Review new file contents in the IDE too. No files staged.'
    }
    elseif ($Action -eq 'Stage') {
        git add -- $migrationFiles
        if ($LASTEXITCODE -ne 0) { throw 'Staging failed' }
        git diff --cached --check
        if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed' }
        git diff --cached --stat
        git diff --cached
        if ($LASTEXITCODE -ne 0) { throw 'Staged review failed' }
    }
    elseif ($Action -eq 'Commit') {
        $stagedPaths = @(git diff --cached --name-only)
        if ($LASTEXITCODE -ne 0) { throw 'Cannot inspect index' }
        if ($stagedPaths.Count -eq 0) { throw 'Nothing staged' }
        if (@($stagedPaths | Where-Object { $_ -notin $migrationFiles }).Count -gt 0) {
            throw 'Unrelated staged files: review manually before committing'
        }
        git commit -m 'research: prepare supervised Colab stability migration (D-020)'
        if ($LASTEXITCODE -ne 0) { throw 'Commit failed' }
        git rev-parse HEAD
        git status --short
    }
    elseif ($Action -eq 'Package') {
        if ($ApprovedHead -notmatch '^[0-9a-f]{40}$') { throw 'Pass the exact reviewed full commit hash' }
        if ([string]::IsNullOrWhiteSpace($Output)) { throw 'Pass a new absolute external output directory' }
        if (-not [IO.Path]::IsPathRooted($Output)) { throw 'Output must be absolute' }
        if (Test-Path -LiteralPath $Output) { throw 'Output already exists; no overwrite' }
        & $pythonPath -B research/tools/build_colab_transfer.py --approved-head $ApprovedHead --output $Output
        if ($LASTEXITCODE -ne 0) { throw 'Release package verification failed' }
        Get-Content -LiteralPath (Join-Path $Output 'TRANSFER_RECEIPT.json')
    }
}
finally { Pop-Location }

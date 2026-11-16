param([Parameter(Mandatory=$true)][string]$Reason, [switch]$Preview)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$rootPath = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$basePath = Join-Path $rootPath 'artifacts\next5'
$rootHistoryPath = Join-Path $rootPath 'BOTTLENECK_BOARD_HISTORY_NEXT5.json'
function Read-JsonFile([string]$Path) {
    Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Get-Sha([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}
function Write-AtomicJson([string]$Path, $Value, [bool]$Immutable = $false) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not ($absolute.StartsWith($basePath + '\', [StringComparison]::OrdinalIgnoreCase) -or $absolute -eq $rootHistoryPath)) {
        throw 'Only owned NEXT5 metadata paths are writable'
    }
    if ($Immutable -and [IO.File]::Exists($absolute)) { throw 'Immutable review already exists' }
    $serialized = $Value | ConvertTo-Json -Depth 100
    $temporary = $absolute + '.' + [Guid]::NewGuid().ToString('N') + '.tmp'
    [IO.File]::WriteAllText($temporary, $serialized + "`n", (New-Object System.Text.UTF8Encoding($false)))
    if ([IO.File]::Exists($absolute)) { [IO.File]::Replace($temporary, $absolute, $null) }
    else { [IO.File]::Move($temporary, $absolute) }
}
$session = Read-JsonFile (Join-Path $basePath 'SESSION.json')
$account = Read-JsonFile (Join-Path $basePath 'runtime\ACCOUNTING.json')
if ($session.status -ne 'ACTIVE_RESEARCH' -or $session.shadow_opened) { throw 'Active unopened research required' }
$age = ([DateTimeOffset]::UtcNow - [DateTimeOffset]$account.timestamp).TotalSeconds
if ($age -lt 0 -or $age -gt 60) { throw 'Fresh real supervisor observation required' }
if (-not $Preview -and $account.current_job -ne 'LATENCY_FIXED_ARTIFACT_BLOCKS') { throw 'Use the original Python reviewer outside isolated timing' }
if (-not $Preview) {
    $policy = Read-JsonFile (Join-Path $basePath 'bottlenecks\TIMING_REVIEW_POLICY_LOCK.json')
    if ($policy.source_sha256 -ne (Get-Sha $PSCommandPath)) { throw 'Native reviewer source changed after freeze' }
}
$boardPath = Join-Path $basePath 'bottlenecks\BOARD.json'
$previousSha = Get-Sha $boardPath
$record = Read-JsonFile $boardPath
if (-not $record.finite_preregistered_menu_complete) { throw 'Only the already closed scientific menu may be snapshotted' }
if ($record.budget.screen_count -ne 20 -or $record.budget.full_count -ne 19) { throw 'Unexpected candidate budget' }
$definitions = Read-JsonFile (Join-Path $basePath 'integrity_audits\fold_summary_definitions_v1\RESULTS.json')
if ($definitions.status -ne 'PASS') { throw 'Scalar provenance required' }
foreach ($entry in $definitions.inputs_sha256.PSObject.Properties) {
    if ((Get-Sha (Join-Path $rootPath $entry.Name)) -ne $entry.Value) { throw ('Scientific scalar source changed: ' + $entry.Name) }
}
$stamp = [DateTimeOffset]::UtcNow.ToString('o')
$record.created_utc = $stamp
$record.review_reason = $Reason
$record.observed_productive_research_seconds = $account.productive_research_seconds
foreach ($row in $record.rows) { $row.timestamp = $stamp }
if (@($record.rows | Where-Object {$_.status -eq 'ACTIVE'}).Count -ne 2) { throw 'Preserve the two active bottlenecks' }
$sourceSha = Get-Sha $PSCommandPath
$record.source_sha256 = $sourceSha
$record | Add-Member -NotePropertyName review_engine -NotePropertyValue 'Native PowerShell metadata review during fixed latency; no Python process or numerical re-estimation' -Force
$record | Add-Member -NotePropertyName review_source_file -NotePropertyValue 'scripts/review_next5_bottlenecks_during_timing.ps1' -Force
$record | Add-Member -NotePropertyName previous_board_sha256 -NotePropertyValue $previousSha -Force
$record | Add-Member -NotePropertyName scientific_scalar_provenance_sha256 -NotePropertyValue (Get-Sha (Join-Path $basePath 'integrity_audits\fold_summary_definitions_v1\RESULTS.json')) -Force
$record | Add-Member -NotePropertyName inherited_priorities_and_scientific_results_unchanged -NotePropertyValue $true -Force
$blocks = @(Get-ChildItem -LiteralPath (Join-Path $basePath 'latency\fixed_v1') -Filter 'repeat_*.json' -ErrorAction SilentlyContinue | ForEach-Object {
    $value = Read-JsonFile $_.FullName
    if ($value.actual_continuous_points -ne 10000000) { throw 'Only complete fixed timing blocks may be summarized' }
    [ordered]@{file=$_.Name; sha256=(Get-Sha $_.FullName); model=$value.model; repeat=$value.repeat; points=$value.actual_continuous_points}
})
$record | Add-Member -NotePropertyName completed_timing_blocks_at_review -NotePropertyValue $blocks -Force
$roundTrip = ($record | ConvertTo-Json -Depth 100) | ConvertFrom-Json
if ($roundTrip.best_measured_full.mean_delta -ne $record.best_measured_full.mean_delta) { throw 'Numerical metadata changed during serialization' }
if ($Preview) {
    [ordered]@{status='PREVIEW_ONLY'; current_job=$account.current_job; previous_board_sha256=$previousSha; completed_timing_blocks=$blocks.Count; source_sha256=$sourceSha; files_written=0; Python_processes_started=0} | ConvertTo-Json
    exit 0
}
$historyPath = Join-Path $basePath 'bottlenecks\history'
$number = @(Get-ChildItem -LiteralPath $historyPath -Filter '*.json').Count + 1
Write-AtomicJson (Join-Path $historyPath ('{0:D3}_REVIEW.json' -f $number)) $record $true
Write-AtomicJson $boardPath $record
Write-AtomicJson (Join-Path $basePath 'bottlenecks\CURRENT.json') $record
$history = @(Get-ChildItem -LiteralPath $historyPath -Filter '*.json' | Sort-Object Name | ForEach-Object { Read-JsonFile $_.FullName })
Write-AtomicJson $rootHistoryPath ([ordered]@{updated_utc=$stamp; reviews=$history})
Write-AtomicJson (Join-Path $basePath 'runtime\ACTIVITY.json') ([ordered]@{timestamp=$stamp; pid=$PID; task='Bottleneck review during isolated timing'; detail=$Reason; phase='RESEARCH'; counts_as_research=$true})
Write-Output ('NEXT5 NATIVE TIMING REVIEW ' + $number + '; complete timing blocks ' + $blocks.Count + '; no Python process started')

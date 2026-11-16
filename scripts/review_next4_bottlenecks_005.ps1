$ErrorActionPreference = 'Stop'
$next4Root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$next4Base = Join-Path $next4Root 'artifacts\next4'
$next4BoardPath = Join-Path $next4Base 'bottlenecks\BOARD.json'
$next4History = Join-Path $next4Base 'bottlenecks\history'
if (@(Get-ChildItem -LiteralPath $next4History -Filter '*.json').Count -ne 4) { throw 'Expected four existing board reviews' }
$next4Board = Get-Content -LiteralPath $next4BoardPath -Raw | ConvertFrom-Json
foreach ($next4Item in $next4Board.bottlenecks) {
    switch ($next4Item.id) {
        'B2' {
            $next4Item.next_test = 'B1 ALT 3/3 positive and H overall safe, but H SCALE delta -0.00347768 crosses zero. Check fixed ARCH/heavy-tail decreases. A2/C3/D2/D3 qualify for conditional unchanged ALT/H only after the frozen synthetic decrease gates. Prefer original SCALE CI lower bound only among fully validated candidates.'
            $next4Item.evidence += @('artifacts/next4/robustness/B1_SPLIT_ENERGY_BRANCHES/RESULTS.json','artifacts/next4/robustness_forensic/B1_SPLIT_ENERGY_BRANCHES/RESULTS.json','artifacts/next4/diagnostic_policy/CONDITIONAL_ROBUSTNESS_VALIDATION.json')
        }
        'B5' {
            $next4Item.next_test = 'Finish isolated direct timing and all 12 already nominated fitted synthetic suites. Explain long-null accumulation with fixed 1024-point local risk sets and paired outlier recovery; no changes to models, seeds or thresholds. Inspect absorbing D3 hazard prior pressure separately from actual fitted-model evidence.'
            $next4Item.evidence += @('artifacts/next4/runtime_validation/B2_DIRECT/RESULTS.json','scripts/diagnose_next4_memory.py','src/next4/sequential.py')
        }
        'B3' {
            $next4Item.status = 'CLOSED_NO_ADMITTED_THIRD_EXPERT'
            $next4Item.solvability = 0.05
            $next4Item.next_test = 'F1 duplicates existing 72/18/10 blends. D2 is the only independent expert above the +0.0003 point threshold but fails its positive CI/probability requirements. F2 is conditionally skipped with zero new fit or identity; no replacement weight search.'
            $next4Item.evidence += @('artifacts/next4/families/F_SELECTION.json')
        }
        'B4' {
            $next4Item.next_test = 'All 14 original primary intervals include zero. The fixed 14-model centered-max diagnostic has no positive joint lower bound. B2 remains positive after every one-series deletion but its small numeric margin is sensitive; this does not replace its original failed CI.'
            $next4Item.evidence += @('artifacts/next4/joint_uncertainty/RESULTS.json','artifacts/next4/influence/B2_LOG_ENERGY_SEMITAIL/RESULTS.json')
        }
        'B7' {
            $next4Item.status = 'DIAGNOSED_LATE_GAIN_EARLY_LOSS'
            $next4Item.next_test = 'B2 time >=200 contributes +0.00151536 against total +0.00130367, with early losses. Report actual scoring-positive support: 182 series at time 0-9, not all 3944 eventual-break memberships. No early-weight tuning in this closed menu.'
            $next4Item.evidence += @('artifacts/next4/scored_support/RESULTS.json','artifacts/next4/influence/B2_LOG_ENERGY_SEMITAIL/TIME_PAIR_CREDIT_SUMMARY.csv')
        }
        'B11' {
            $next4Item.next_test = 'G1 H context consumes 66-68 percent of split gain in the failed screen, a descriptive fitted-structure observation. E1/G2 SCALE intervals are strictly negative; no additional learned-meta robustness fits or feature ablation search.'
            $next4Item.evidence += @('artifacts/next4/closed_menu_diagnostics/RESULTS.json')
        }
    }
    $next4Item.priority = $next4Item.expected_impact * $next4Item.evidence_strength * $next4Item.solvability / ((1+$next4Item.overfit_risk)*(1+$next4Item.compute)*(1+$next4Item.complexity))
}
$next4Items = @($next4Board.bottlenecks | Sort-Object priority -Descending)
if (@($next4Items | Where-Object status -eq 'ACTIVE').Count -gt 2) { throw 'More than two active bottlenecks' }
for ($next4Index=0; $next4Index -lt $next4Items.Count; $next4Index++) { $next4Items[$next4Index].rank=$next4Index+1 }
$next4Reason = 'B1/B2 fixed ALT/H and causal evidence are complete; primary performance remains unpromoted. Five supported real SCALE candidates remain contingent on fixed synthetic and runtime requirements. Focus on mechanism transfer and fitted-memory stability. Board metadata updated in native PowerShell to preserve the isolated Python timing process.'
$next4Time = [DateTimeOffset]::UtcNow.ToString('o')
$next4Result = [ordered]@{ created_utc=$next4Time; reason=$next4Reason; maximum_active=2; bottlenecks=$next4Items; priority_scope='Subjective decision heuristic, not a measured or attainable TS-AUC gain' }
$next4Json = $next4Result | ConvertTo-Json -Depth 12
$next4Immutable = Join-Path $next4History '005.json'
if (Test-Path -LiteralPath $next4Immutable) { throw 'Refusing to overwrite a review' }
[IO.File]::WriteAllText($next4Immutable,$next4Json,(New-Object Text.UTF8Encoding($false)))
[IO.File]::WriteAllText($next4BoardPath,$next4Json,(New-Object Text.UTF8Encoding($false)))
$next4Rows = foreach ($next4Item in $next4Items) {
    [pscustomobject][ordered]@{ timestamp=$next4Time; review=5; id=$next4Item.id; status=$next4Item.status; priority=$next4Item.priority; reason=$next4Reason; next_test=$next4Item.next_test }
}
$next4Rows | Export-Csv -LiteralPath (Join-Path $next4Base 'bottlenecks\HISTORY.csv') -NoTypeInformation -Append -Encoding UTF8
Write-Output 'NEXT4 bottleneck review 005 complete; no additional Python process launched.'

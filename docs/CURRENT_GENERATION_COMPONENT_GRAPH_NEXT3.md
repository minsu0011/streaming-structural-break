# Scientific model and runtime boundaries

The frozen 80/20 blend remains the performance reference. New scientific recipes are evaluated separately; no recipe has been promoted at this intermediate checkpoint. The exact runtime branch is a deployment implementation of the same scientific model.

```mermaid
flowchart LR
  H[Known historical observations H] --> A[H-fitted AR8 and ARCH1 parameters]
  H --> L[Legacy historical calibration]
  X[One current online observation] --> S[Shared causal update]
  A --> S
  L --> JF[Selected legacy features]
  X --> JF
  S --> VF[Two serial variance channels]
  S --> DF[Four lag products and six evidence statistics]
  JF --> J[J102 frozen trees]
  VF --> J
  DF --> C[C205 frozen trees]
  J --> B[Float32 component scores; fixed 80/20 blend]
  C --> B
  B --> O[One probability for the current point]
  H --> PC[H-dependent serial constants]
  AG[Fixed age grid] --> DC[Shared read-only dependence constants]
  PC -. exact arithmetic optimization .-> VF
  DC -. exact arithmetic optimization .-> DF
```

There is no future-suffix, total online-length, dataset-ID or mechanism-label input to this graph. Training and post-outcome diagnostics can use labels and group identities under their separate DEV guards. The first public generator output is `None` before the dataset iterator is touched.

```mermaid
flowchart TB
  DEV[8,000 allowed DEV series] --> OUTER[Original outer group split]
  OUTER --> TR[Outer training groups]
  OUTER --> VA[Outer validation groups]
  TR --> INNER[Original group labels reused for inner splits]
  INNER --> BASE[Fit J102 and C205 inside inner training only]
  BASE --> IOOF[Base predictions on inner validation groups]
  IOOF --> G[Fit one registered gate or residual rule]
  TR --> OB[Outer base models fitted without outer validation]
  OB --> OP[Outer validation base predictions]
  OP --> APPLY[Apply the fixed fitted gate or residual]
  G --> APPLY
  VA --> SCORE[Official TS-AUC and diagnostic labels]
  APPLY --> SCORE
```

No outer-validation observation or target enters inner base training or meta-model fitting. The checkpoint audit reconstructs all 56 nested base-fit coordinate and target hashes. This prevents the specific stacking leakage under review; it does not undo earlier adaptive selection of the frozen base recipes on DEV.

For the final full-DEV R3 gate, all three leaves round to C205 weight 0.1. Its deployed behavior is therefore constant 90/10, despite being produced by a depth2 training procedure. The original outer-CV gates remain separately fitted, and their held-out scores are never replaced with scores from this full-DEV gate.

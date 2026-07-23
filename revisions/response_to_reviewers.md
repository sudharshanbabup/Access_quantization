# IEEE Access - Revision Response & Verification Data

This folder contains the responses to each reviewer's comment and the corresponding empirical results, verification data, and files.

---

## Major Weaknesses

### 1. Theory Assumptions are Too Strong
*   **Comment:** The theorems assume smoothness, Lipschitz continuity, fixed activation regions, and small perturbations, but there is no discussion of when these assumptions fail.
*   **Response:** We have added a new subsection **Section III-D: Practical Validity of Assumptions** in [sec_theory.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_theory.tex) that details the limits of these assumptions on modern CNN components (ReLU activations, BatchNorm folding, residual connections, and self-attention in ViTs).

---

### 2. Proof Sketches
*   **Comment:** Several proofs in the main text are sketches rather than complete proofs, which reduces mathematical rigor.
*   **Response:** We created a new Appendix section in [sec_appendix.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_appendix.tex) that provides the detailed, step-by-step algebraic proof of Theorem 1 (gradient perturbation bound) and added it to [main.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/main.tex).

---

### 3. Comparisons with SOTA PTQ Methods
*   **Comment:** Need comparisons against modern PTQ methods like AdaRound and BRECQ.
*   **Response:** We added a comparative analysis in **Section V-D: Deployment Metrics and SOTA Comparisons** in [sec_results.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_results.tex). EAQ directly minimizes explanation distortion in closed form ($\sim 45$ seconds) and achieves superior explanation fidelity compared to accuracy-optimized PTQ methods like BRECQ and AdaRound, which take hours to solve parameter rounding.
*   **Verification Data:** [SOTA PTQ Comparison Table](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_3_comparisons/sota_comparison.csv)

---

### 4. Runtime and Latency Analysis
*   **Comment:** Reviewers expect deployment metrics: Latency, Memory, Energy, Model Size.
*   **Response:** We measured inference latency, parameter storage size (KB), and relative energy consumption across quantization bit-widths for both CIFAR-10 and ImageNette, added as Table V in [sec_results.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_results.tex).
*   **Verification Data:** [Runtime Metrics CSV](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_4_runtime/runtime_metrics.csv)

---

### 5. Computational Complexity Analysis
*   **Comment:** Need a computational complexity table (Big-O analysis for sensitivity, allocation, memory, FLOPs).
*   **Response:** We added a detailed complexity table and discussion in **Section IV-D: Computational Complexity** in [sec_eaq.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_eaq.tex) comparing EAQ against Uniform, AdaRound, BRECQ, and ECQ.
*   **Verification Data:** [Complexity Table CSV](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_5_complexity/complexity.csv)

---

### 6. Datasets
*   **Comment:** SynthShapes-32 is synthetic; need natural-image evaluations.
*   **Response:** We highlight and expand the discussion of the natural-image validation benchmarks on **CIFAR-10** (baseline test accuracy of **81.42%** using the scaled ResNet backbone with 302k parameters) and **ImageNette** (baseline accuracy of **95.95%** using a standard ResNet-18 model).

---

### 7. Explanation Metrics (Faithfulness)
*   **Comment:** Broaden metrics to include standard XAI evaluation metrics like Insertion and Deletion AUC.
*   **Response:** We added `insertion_auc` to the attribution evaluation suite in [code/attributions.py](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/code/attributions.py) and ran Insertion vs Deletion AUC analysis.
*   **Verification Data:** [Insertion/Deletion AUC CSV](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_7_metrics/insertion_deletion_auc.csv)

---

### 8. Statistical Confidence
*   **Comment:** Report statistical confidence / 95% confidence intervals.
*   **Response:** We run Bootstrap resampling ($B=1000$ trials) to report the 95% confidence intervals for the Spearman and Pearson explanation fidelity gains of EAQ over Uniform quantization.
*   **Verification Data:** [Bootstrap Confidence Intervals CSV](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_8_confidence/bootstrap_confidence_intervals.csv)

---

### 9. Ablation Study
*   **Comment:** Need ablation on the sensitivity scoring and layer bit-allocation of EAQ.
*   **Response:** We added **Section VI-A: EAQ Ablation Study** in [sec_discussion.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_discussion.tex) and compared EAQ bit allocations against Random and parameter Magnitude-based allocations.
*   **Verification Data:** [Ablation Study CSV](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/revisions/weakness_9_ablation/ablation_study.csv)

---

### 10. Failure Cases
*   **Comment:** Show failure analysis and limitations of EAQ.
*   **Response:** We added **Section VI-B: Limitations and Failure Cases** in [sec_discussion.tex](file:///Users/sudharshanbabupandava/JioCloud/CMR%20University/Research/Ravi%20Saidala/Antigravity_12/paper/sec_discussion.tex) detailing low-bit collapse ($b \le 2$), calibration gradient noise, and ill-conditioned networks.

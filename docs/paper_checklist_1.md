# NeurIPS Paper Checklist Guidelines

The NeurIPS Paper Checklist is designed to encourage reproducibility, transparency, responsible machine learning practice, research ethics, and awareness of societal impact.

The checklist is included in the LaTeX style file. Do not remove it. Papers without the checklist will be desk rejected.

All submissions must be a **single PDF** in this order:

1. main paper
2. optional technical appendix
3. NeurIPS paper checklist

The checklist must follow references and optional supplemental material. It does **not** count toward the page limit.

Please read the checklist guidance carefully.

For each question, answer **yes**, **no**, or **n/a**.

`NA` means:

- **N/A**: the question is not applicable to this paper, or
- **N/A**: the relevant information is not available.

You may add 1-2 sentences of justification when needed, and ideally point to the relevant paper section(s).

Checklist answers are visible to reviewers and become part of the final publication.

## 1. Claims

**Question:** Do the main claims in the abstract and introduction accurately reflect the paper's contributions and scope? (including generalizability and assumptions)

The paper's contributions and key assumptions should be clear in the abstract and introduction, and they should align with the theory and experiments. Aspirational goals are acceptable if clearly framed as not yet achieved.

## 2. Limitations

**Question:** Do the authors discuss the limitations of their approach?

The paper should include a separate limitations section (or equivalent) that explicitly states assumptions, robustness to assumption violations, and conditions under which performance may degrade. Authors should also describe scope-related limits (e.g., number of datasets, number of runs, environmental factors).

It is better to be transparent about limits than to omit them.

## 3. Theory, Assumptions, and Proofs

**Question:** If presenting theoretical results, did you state all assumptions and include complete proofs?

- State all assumptions clearly, ideally within theorem statements or explicit references.
- Include complete proofs (in-paper or supplement).
- If proofs are in supplemental material, consider including a short intuition/proof sketch in the main text.
- Clearly reference related theorems, lemmas, and literature.
- Informal proofs in the main text should be supported by formal proofs in supplemental material.

## 4. Experimental Result Reproducibility

**Question:** If the contribution is a dataset or model, what did you do to make results reproducible or verifiable?

Reproducibility can be achieved by releasing code/data, model access, detailed replication instructions, checkpoints, or another method appropriate for the contribution. If code is not released, a reasonable alternative path for replication should still be described.

## 5. Open Access to Data and Code

**Question:** If you ran experiments, did you include code, data, and instructions to reproduce results?

- Reproducibility should include required commands, environment setup, and where to find code/data.
- This can be in supplemental material or via URL.
- "No" is acceptable when justified (for example, proprietary constraints), but reviewers should not reject solely on this basis unless central to the contribution.
- Preserve anonymity at submission time.

## 6. Experimental Setting and Details

**Question:** If you ran experiments, did you specify all training details (data splits, hyperparameters, selection procedure)?

- Key experimental details should be in the main paper.
- Full details may be supplied in code, supplemental material, or an appendix.
- Clearly explain how hyperparameters were chosen.

## 7. Experimental Statistical Significance

**Question:** Did the paper report error bars or other evidence of statistical significance?

If experiments are included, authors should:

- report error bars/confidence intervals/significance tests for claims, and
- specify what variability is captured,
- describe computation method, and
- state assumptions.

Also report whether bars are standard deviation or standard error. For asymmetric quantities, avoid symmetric bars that imply impossible values.

## 8. Experiments Compute Resource

**Question:** Do you provide enough compute details to reproduce experiments?

Include at minimum:

- hardware (CPU/GPU, cluster/cloud),
- memory/storage,
- compute time per run,
- total compute used,
- whether omitted runs (e.g., failures or preliminary experiments) required additional compute.

## 9. Code of Ethics

**Question:** Did you review and follow the NeurIPS Code of Ethics?

If any deviations are needed, explain them clearly.

## 10. Broader Impacts

**Question:** If appropriate, did you discuss potential negative societal impacts?

Discuss potential harms and risks, including misuse scenarios, fairness/privacy/security concerns, and deployment-context risks. Where relevant, include mitigation strategies.

## 11. Safeguards

**Question:** For high-risk or potentially misused models, are safeguards in place for controlled release?

Examples include usage restrictions, access control, or licensing/usage agreements for release.

## 12. Licenses

**Question:** If using existing assets (code, data, models), did you cite creators and respect licenses?

Include:

- source citations and version,
- relevant URLs,
- explicit license names (e.g., CC-BY 4.0),
- copyright and terms of service for scraped sources,
- original and derived licenses when repackaging datasets/assets.

If license details are unavailable, contact asset creators.

## 13. Assets

**Question:** If releasing new assets, did you document them fully?

Published assets should include structured details on training, limits, license, intended use, and other submission requirements. Include any required consent details and anonymize assets where needed.

## 14. Crowdsourcing and Research with Human Subjects

**Question:** If using crowdsourcing/human subjects, did you include instructions, screenshots, and compensation details where applicable?

At minimum, provide participant instructions and payment methodology for core human-subject studies.

## 15. IRB Approvals

**Question:** If applicable, did you describe participant risks and IRB (or equivalent) review?

If approved, state this clearly. Do not include identifying institutional details in initial submissions if that breaks anonymity.

## 16. Declaration of LLM Usage

**Question:** Does the paper describe LLM usage when it is an important, original, or non-standard part of the method?

LLM use for writing/editing only (without methodological impact) does not require declaration. If core methods rely on LLMs in a substantial way, include a clear explanation.

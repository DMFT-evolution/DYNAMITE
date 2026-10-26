# Cite

If you use DYNAMITE, please cite the paper that describes the code:

- J. Lang, V. Citro, L. Leuzzi, F. Ricci-Tersenghi, “DYNAMITE: A high-performance framework for solving Dynamical Mean-Field Equations,” [arXiv:2604.06309](https://arxiv.org/abs/2604.06309) (under review).

Where you refer specifically to the original idea of the method, you may in addition cite

- J. Lang, S. Sachdev, S. Diehl, “Numerical renormalization of glassy dynamics,” Phys. Rev. Lett. **135**, 247101 (2025), [doi:10.1103/z64g-nqs6](https://doi.org/10.1103/z64g-nqs6), [arXiv:2504.06849](https://arxiv.org/abs/2504.06849).

That paper was written for a much older version of the code; for the algorithm as implemented here, refer to the first one.

For reproducibility, also state the version of the program you used. `./RG-Evo -v` prints it, and every run records it, together with the Git commit, in its `params.txt`. Machine-readable citation data are in [`CITATION.cff`](https://github.com/DMFT-evolution/DYNAMITE/blob/main/CITATION.cff) in the repository; GitHub and reference managers such as Zotero can read this file.

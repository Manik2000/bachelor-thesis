# Bachelor thesis

This repo contains my bachelor thesis (written in Polish) titled "Selected clustering ensemble methods" ("Wybrane metody grupowania zespołowego" accordingly).
Besides the pdf file with thesis, I also uploaded Python scripts and Jupyter notebooks with the most important methods implementations, and analysis performing code.

---
In the `scripts` directory there are:
* `algorithms.py` containing implementations of ensemble clustering methods that were chosen and described in the thesis,
* `evaluation.py` implementing important clustering evaluation metrics that are not present in `sklearn`,
* `my_own.py` with code generating artificial dataset used for evaluation.

In the `notebooks` directory there are in contrast:
* `Case_study.ipynb` with analysis from the second part of the thesis,
* `analysis.ipynb` in which main results are generated,
* `diversity.ipynb` containing analysis on the impact of diversity of the ensembles on the stability and quality of the results,
* `hyperparameters_tuning.ipynb` with analysis on tuning methods' hyperparameters.

Python package with methods from the thesis and other is available [here](https://github.com/Manik2000/ensemble-clustering).

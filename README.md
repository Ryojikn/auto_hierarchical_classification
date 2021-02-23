# Auto Hierarchical Classification

This implementation is created for multiclass NLP problem.

And its my own implementation of the concepts provided in the paper below:


> Daniel Silva-Palacios, Cèsar Ferri, María José Ramírez-Quintana,
> **Improving Performance of Multiclass Classification by Inducing Class Hierarchies,**
> Procedia Computer Science,
> Volume 108,
> 2017,
> Pages 1692-1701,
> ISSN 1877-0509,
> https://doi.org/10.1016/j.procs.2017.05.218.
> (https://www.sciencedirect.com/science/article/pii/S1877050917308244)'''


## How it works?

The concept provided by Palacios et al. explains how to create a similarity matrix based on confusion matrix predicted by the model, and then group the most "confused" targets.

By applying that, I was able to improve the overall results for the text classifier model.

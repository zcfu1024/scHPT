# scHPT
Test for detecting and explaining heterogeneous effects in single-cell perturbation data
# Method Overview
<img src="method.png" width="800">
scHPT consists of three hypothesis test.

Test 1 evaluates whether perturbation effects are homogeneous across all cells. 

Test 2 considers discrete covariates, such as cell types or cell lines, and tests whether variation across cell subgroups explains the full variation in perturbation effects. This allows identification of specific subpopulations exhibiting heterogeneous responses. 

Test 3 extends this idea to continuous covariates, such as time or dosage, by testing whether perturbation effects can be captured by a covariate-dependent effect.

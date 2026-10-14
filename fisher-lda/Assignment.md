Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

# **Introduction to Machine Learning** 

Assignment 1: Fisher’s Linear Discriminant Analysis 

September 2026 

**Total: 30 points** _Take-home submission + live presentation/viva_ 

**Core task.** Reconstruct Fisher’s linear discriminant analysis on the classic Iris measurements using basic NumPy linear algebra, without using a library implementation of LDA. Compute the scatter matrices, solve the generalized eigenvalue problem, obtain the two discriminant directions, project the observations into two dimensions, and compare the results with Fisher (1936). 

**Viva eligibility.** Only students who submit a valid assignment by the submission deadline are eligible for the live presentation/viva. **No submission, no viva.** 

## **1 Background** 

R. A. Fisher’s 1936 paper, _The Use of Multiple Measurements in Taxonomic Problems_ , studies how several measurements can be combined into linear functions that discriminate between populations. The paper uses measurements of Iris flowers and develops discriminating linear functions from observed means and within-species sums of squares and products. 

Fisher first works explicitly with two species, _Iris setosa_ and _Iris versicolor_ , and later considers the three-species case including _Iris virginica_ . The original paper therefore provides a useful historical reference for both the two-class discriminant calculation and the multi-class extension. 

Your task is to implement the central calculation yourself and then explain the linear algebra behind it. 

## **2 Data** 

Use the classic Iris dataset with 150 observations, 50 from each of three species, and four flower measurements: 

- (a) sepal length, 

- (b) sepal width, 

- (c) petal length, 

- (d) petal width. 

Fisher’s original paper contains the underlying measurements in Table I. You may use a machinereadable copy of the standard Iris dataset for implementation, but verify the observations, class labels, and feature definitions against the original source. 

1 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 



Figure 1: Illustrative view of Iris observations in the original measurement space. This is not an LDA projection. 

## **3 Implementation Restriction** 

**Restriction for the core calculation:** Do not use a library LDA routine to obtain the discriminant directions. 

In particular, do not use: 

- scikit-learn’s `LinearDiscriminantAnalysis` ; 

- MATLAB’s built-in discriminant-analysis functions; 

- R’s `lda` routine; 

- another ready-made LDA implementation. 

You may use NumPy for array operations and numerical linear algebra, including matrix multiplication, linear solves, eigendecomposition, and norms. You may use Pandas for data handling and Matplotlib (or an equivalent plotting package) for visualisation. 

A library LDA implementation may be used only as a **post-hoc check** after your own implementation is complete. If used, clearly label it as verification and do not use its output as the basis of your solution. 

## **4 Mathematical Setup** 

Let an observation be 



and let there be _C_ = 3 classes. Let _µk_ denote the mean vector of class _k_ and _µ_ the overall mean. The within-class scatter matrix is 



2 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

The between-class scatter matrix is 



where _nk_ is the number of observations in class _k_ . 

For a direction _w_ , Fisher’s criterion is 



Maximising this criterion leads to the generalized eigenvalue problem 



For _C_ classes, there are at most _C −_ 1 useful discriminant directions. Therefore, for the three Iris species, at most two dimensions are required to represent the discriminant subspace. Because 



at most two generalized eigenvalues can be nonzero for this three-class problem, up to floating-point and conditioning effects. 

## **5 Required Tasks** 

### **5.1 Task 1 – Load, Inspect, and Document the Data** 

1. Load the Iris dataset. 

2. Report its shape. 

3. Identify the four measurements and three species. 

4. Verify that each class contains 50 observations. 

5. Compute the mean vector of each class. 

6. Compute the overall mean vector. 

Include a compact table of the three class means. 

### **5.2 Task 2 – Reconstruct Fisher’s Two-Species Calculation** 

Before moving to the full three-class formulation, reproduce the key idea in Fisher’s original setosa-versicolor calculation. 

For this task, use **only** the observations from _Iris setosa_ and _Iris versicolor_ . Let 



Construct the within-class scatter matrix using these two classes and solve 



Then: 

1. report the resulting coefficient vector _a_ ; 

3 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

2. rescale the vector so that the sepal-length coefficient is 1; 

3. compare the resulting coefficient ratios with Fisher’s reported two-species linear compound; 

4. explain any numerical differences. 

Fisher reports, after choosing the coefficient of sepal length to be unity, the approximate two-species linear function 



**Do not simply copy these numbers.** Obtain your coefficients from your own calculation and then compare them with Fisher’s result. 

### **5.3 Task 3 – Compute the Within-Class Scatter Matrix** 

For all three species, implement 



You must: 

1. show the mathematical expression implemented; 

2. report the resulting 4 _×_ 4 matrix; 

3. verify numerically that _SW_ is symmetric up to numerical precision. 

### **5.4 Task 4 – Compute the Between-Class Scatter Matrix** 

Implement 



Report: 

1. the resulting 4 _×_ 4 matrix; 

2. its numerical rank; 

3. an explanation of why its rank is at most _C −_ 1. 

### **5.5 Task 5 – Solve the Generalized Eigenvalue Problem** 

Solve 



A direct NumPy implementation may use 



but **prefer a linear solve such as** `np.linalg.solve(SW, SB)` **over explicitly computing the inverse** . 

You must: 

1. compute all eigenvalues and eigenvectors; 

4 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

2. sort the eigenpairs by decreasing eigenvalue; 

3. retain the two eigenvectors corresponding to the two largest eigenvalues; 

4. report the eigenvalues and the two retained directions; 

5. compute and report the generalized-eigenvalue residual for each retained direction: 



For three classes, the remaining discriminant eigenvalues should be zero or numerically close to zero, up to floating-point and conditioning effects. 

### **5.6 Task 6 – Project the Iris Observations into Two Dimensions** 

Form 



and project each observation using 



This gives the mapping 



Create a two-dimensional scatter plot of all 150 projected observations. Clearly identify the three Iris species and label both axes. 



Figure 2: Example layout only. The values shown are illustrative and must not be reported as experimental results. Your submission must contain a new figure generated from your own NumPy implementation. 

### **5.7 Task 7 – Compare Your Result with Fisher (1936)** 

Read the relevant numerical parts of Fisher’s paper, including the discussion of the two-species discriminant and the later three-species construction. 

Address the following: 

1. How do your class means compare with Fisher’s reported means? 

5 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

2. How do your two-class coefficient ratios compare with 



3. Fisher’s later three-species construction reports a linear compound with coefficients approximately 

   - ( _−_ 3 _._ 308998 _, −_ 2 _._ 759132 _,_ 8 _._ 866048 _,_ 9 _._ 392551) 

in the order (sepal length, sepal width, petal length, petal width). How does this relate to your multi-class discriminant analysis? 

4. Why need the numerical coefficients not match exactly even when the discriminant direction is effectively the same? 

5. What do sign and scale ambiguities of eigenvectors mean in practice? 

6. Does your two-dimensional discriminant projection provide qualitative separation consistent with Fisher’s analysis? 

The paper’s reported coefficients are included only to make the comparison precise; students must still perform the calculation and explain the relationship. 

### **5.8 Task 8 – Interpretation** 

Answer briefly and precisely: 

1. Why does Fisher’s criterion contain within-class scatter in the denominator? 

2. Why are there at most two useful discriminant directions for three classes? 

3. What does a large generalized eigenvalue indicate? 

4. Which original measurements have the largest absolute coefficients in each retained direction? How should those coefficients be interpreted? 

5. What information is discarded by reducing the observations from four dimensions to two? 

## **6 Numerical Verification** 

Your submission must include at least the following checks: 

1. symmetry check for _SW_ ; 

2. symmetry check for _SB_ ; 

3. numerical rank of _SB_ ; 

4. generalized-eigenvalue residuals for the two retained directions; 

5. a check of the dimensions of all matrices and vectors; 

6. a consistency check showing that rescaling or sign-flipping an eigenvector does not change its discriminant axis. 

For the symmetry checks, report numerical quantities such as 



For the eigenvector sign/scale check, demonstrate numerically that replacing _w_ by _cw_ for a nonzero scalar _c_ produces the same one-dimensional subspace. 

6 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

## **7 Optional Viva Extensions** 

The following are **not required additions to the written submission** , but you may be asked about them during the live viva: 

1. What happens if _SW_ is singular or ill-conditioned? 

2. Why might standardising the measurements change the discriminant directions? 

3. How would unequal class sizes affect _SB_ ? 

4. Why is the two-class discriminant based on _SW_<sup>_−_1</sup><sup>_d_?</sup> 

5. How would you verify your implementation without using sklearn’s LDA implementation? 

6. Why can two numerically different coefficient vectors represent the same discriminant axis? 

7. What information is lost in the two-dimensional projection? 

8. How would you classify a new observation after obtaining the discriminant subspace? 

9. How would you test whether the observed separation is robust to resampling the dataset? 

10. What numerical issues might arise from solving the generalized eigenvalue problem through _SW_<sup>_−_1</sup><sup>_SB_?</sup> 

## **8 What You May and May Not Use** 

#### **Allowed:** 

- Python; 

- NumPy; 

- Pandas; 

- Matplotlib; 

- Jupyter notebooks; 

- standard matrix operations and eigenvalue routines; 

- the standard Iris dataset; 

- the original Fisher (1936) paper and other cited references. 

**Not allowed for the core calculation:** 

- sklearn’s LDA implementation; 

- MATLAB or R routines that directly return an LDA solution; 

- copying the final discriminant coefficients from Fisher without performing the calculation; 

- a library-generated LDA plot presented as your own implementation. 

LLMs may be used as learning and coding aids. You remain responsible for understanding and verifying all mathematics, code, numerical results, and sources used in the submission. 

## **9 Submission Requirements – 5 Points** 

Submit **one Jupyter notebook (.ipynb)** as the primary submission. 

The notebook must: 

7 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

- run from top to bottom without manual intervention; 

- contain both code and concise written explanations; 

- display the required numerical outputs and figures; 

- use reproducible code and clearly identified data sources. 

The notebook should contain the following sections: 

1. Dataset and class means 

2. Two-species Fisher reconstruction 

3. Three-class within-class scatter 

4. Three-class between-class scatter 

5. Generalized eigenvalue solution 

6. Two-dimensional LDA projection and figure 

7. Numerical verification 

8. Comparison with Fisher (1936) 

9. Interpretation questions 

10. References 

### **5-Point Submission Rubric** 

|**Criterion**|**Points**|
|---|---|
|Complete required tasks and outputs|2|
|Correct NumPy implementation and numerical results|2|
|Clarity, organisation, reproducibility, and citations|1|
|**Total**|**5**|



A notebook that runs only partially, has missing outputs, or relies on hidden or manual calculations will not demonstrate the required implementation. 

## **10 Live Presentation and Viva – 25 Points** 

The take-home submission will be evaluated through a live presentation and viva. The live component is intended to test whether you understand the mathematics, implementation, numerical results, and interpretation of the work you submitted. 

### **10.1 Logistics** 

Each **student** will receive approximately 10 minutes for the live evaluation: 

- approximately 4 minutes to present the problem, method, key calculations, and main figure; 

- approximately 6 minutes for questions and viva discussion. 

**Only students who have submitted the assignment are eligible for the viva. No submission, no viva.** 

8 

Assignment 1: Fisher’s LDA 

Introduction to Machine Learning 

Be prepared to open your notebook and explain or rerun relevant sections of your implementation during the viva. 

The presentation should focus on reasoning and results rather than reading code line by line. 

### **10.2 Viva Rubric** 

|**Component**|**Points**|
|---|---|
|Mathematical derivation and conceptual understanding<br>NumPy implementation and numerical linear algebra|6<br>5|
|Scatter matrices, eigenvalues, eigenvectors, and verifica-|4|
|tion||
|Projection, visualisation, and interpretation|4|
|Comparison with Fisher (1936) and interpretation of<br>differences|4|
|Clarity, independence, and quality of responses|2|
|**Total**|**25**|



## **11 Final Checklist** 

Before submitting, verify: 

□ Iris data loaded and □ _SB_ computed from its □ Generalized eigenproblem documented. definition. solved using NumPy. □ Three class means and overall □ Numerical rank of _SB_ □ Two leading discriminant mean. reported. directions identified. □ Two-species Fisher calculation □ Eigenvalue residuals reported. □ All 150 observations projected reproduced. to 2D. □ _SW_ computed from its □ Eigenvector sign/scale □ Three species shown in final definition. ambiguity explained. 2D plot. □ Symmetry of _SW_ verified. □ Interpretation questions □ Fisher (1936) comparison answered. completed. □ Notebook runs reproducibly. □ All external sources cited. 

- Notebook runs reproducibly. 

## **12 References** 

1. R. A. Fisher, “The Use of Multiple Measurements in Taxonomic Problems,” _Annals of Eugenics_ , 7(2), 179–188, 1936. doi:10.1111/j.1469-1809.1936.tb02137.x. 

_Understand the calculation, not only the final plot._ 

9 


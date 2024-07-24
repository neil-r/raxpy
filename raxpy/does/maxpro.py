"""

Designing Computer Experiments with 
Multiple Types of Factors: The MaxPro Approach  

https://par.nsf.gov/servlets/purl/10199193


I. Initalization Stage:

Step 1. If p1 > 0, generate a n × p1 random LHD Dx for 
continuous factors.

Step 2. If p2 > 0, generate a n × p2 random LHD Du and collapse the n
levels in each column to the nearest given discrete numeric levels of
each factor.
10
Step 3. If p3 > 0, choose a n × p3 optimal design for nominal factors 
Dv from the existing physical experiments’ literature or using a 
commercial statistical software such as JMP. 

Step 4. Form the n × (p1 + p2 + p3) initial design matrix 
D = [Dx, Du, Dv], which consists n levels for each of the p1
continuous factors, mk levels for the kth discrete numeric 
factor (k = 1, . . . , p2), and Lh levels for the hth nominal
factor (h = 1, . . . , p3).

II. Optimization Stage: Iteratively search in the design space to 
optimize the criterion (9) using a version of the simulated annealing
algorithm (Morris and Mitchell 1995).
 
Step 5. Denote the current design matrix as D = [Dx, Du, Dv]. Randomly 
choose a column from the [Dx, Du] components, and interchange two
randomly chosen elements within the selected column. Denote the new 
design matrix as Dtry.

Step 6. If Dtry = D, repeat Step (5).

Step 7. If ψ(Dtry) < ψ(D), replace the current design D with Dtry; 
otherwise, replace the current design D with Dtry with probability
π = exp{−[ψ(Dtry) − ψ(D)]/T }, where T is a preset parameter
known as “temperature”.

Step 8. Repeat Step (5) to Step (7) until some convergence requirements
are met. Report the design matrix with the smallest ψ(D) value as the 
optimal design with respect to criterion (9).
"""

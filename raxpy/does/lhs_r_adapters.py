import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri


def generate_maxpro_design(n: int, d: int, p: int = 2):
    # Activate R to pandas conversion
    pandas2ri.activate()

    # Execute R code to create a table
    robjects.r(
        f"""
        library(MaxPro)
            
        #{n}-run design
        #{d} continuous factors, 1 discrete numeric factor (6 levels)
        #Generate a random Latin hypercube design (each factor standardized into [0,1])
        #as the initial design for continuous factors
        rand_design_part1=apply(matrix(rep(seq(from=0,to=1,length={n}),{d}),ncol={d}),2,sample)
        #Generate a random initial design (standardized into [0,1]) for discrete numeric factors
        rand_design_part2=sample(rep(seq(from=0,to=1,length=6),each=3))
        #Construct an optimal design for the two nominal factors
        #OA_matrix=cbind(rep(1:3,each=6),rep(1:3,6))
        #Initial design matrix
        InitialDesign=cbind(rand_design_part1,rand_design_part2) #,OA_matrix)
        #Optimize the design based on MaxProQQ criterion
        obj=MaxProQQ(InitialDesign, p_nom={p})
        d <- obj$Design
    """
    )

    # Retrieve the R table object
    design = robjects.r["d"]

    return design

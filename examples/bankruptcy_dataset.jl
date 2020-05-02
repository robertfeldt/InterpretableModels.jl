# Based on the bankruptcy dataset downloaded from Berk Ustun:
#  https://github.com/ustunb/slim-python/blob/master/data/bankruptcy_processed.csv
using CSV
using InterpretableModels

df = CSV.read(joinpath(@__DIR__(), "data", "bankruptcy_processed.csv"))

classcolumn = 1
bslim1 = bboptimize_slim_scoring_system(df, classcolumn, 0.1; MaxTime = 3.0, PopSize = 100)
bslim2 = bboptimize_slim_scoring_system(df, classcolumn, 0.01; MaxTime = 10.0, PopSize = 200)
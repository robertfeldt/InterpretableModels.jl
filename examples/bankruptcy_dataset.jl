# Based on the bankruptcy dataset downloaded from Berk Ustun:
#  https://github.com/ustunb/slim-python/blob/master/data/bankruptcy_processed.csv
using CSV
using InterpretableModels

df = CSV.read(joinpath(@__DIR__(), "data", "bankruptcy_processed.csv"))

classcolumn = 1

bslim1   = bboptimize_slim_scoring_system(df, classcolumn, 1; MaxTime = 5.0, PopSize = 200)
bslim01  = bboptimize_slim_scoring_system(df, classcolumn, 0.1; MaxTime = 5.0, PopSize = 200)
bslim001 = bboptimize_slim_scoring_system(df, classcolumn, 0.01; MaxTime = 5.0, PopSize = 200)
using InterpretableModels: slim_scoring_loss01_norm0, column_name_and_index
using DataFrames

@testset "slim_scoring_loss01_norm0" begin
    y = Int[1, -1]
    X = Int[1 0 -1; 0 -1 1]
    λ = [0, 1, 2]
    # sum(y .* (X * λ) .<= 0) but we write it out per element
    @test slim_scoring_loss01_norm0(λ, X, y, 10) ==
        ((y[1] * (X[1, 1] * λ[1] + X[1, 2] * λ[2] + X[1, 3] * λ[3]) <= 0) +
         (y[2] * (X[2, 1] * λ[1] + X[2, 2] * λ[2] + X[2, 3] * λ[3]) <= 0))/2 +
         (10 * sum(λ .!= 0))

    @test slim_scoring_loss01_norm0(λ, X, y, 0.34) ==
         ((y[1] * (X[1, 1] * λ[1] + X[1, 2] * λ[2] + X[1, 3] * λ[3]) <= 0) +
          (y[2] * (X[2, 1] * λ[1] + X[2, 2] * λ[2] + X[2, 3] * λ[3]) <= 0))/2 +
          (0.34 * sum(λ .!= 0))

    @test slim_scoring_loss01_norm0(λ, X, float.(y), 0.34) ==
          slim_scoring_loss01_norm0(λ, X,        y,  0.34)

    @test slim_scoring_loss01_norm0(λ, float.(X), float.(y), 1e-3) ==
          slim_scoring_loss01_norm0(λ,        X,         y,  1e-3)

    @test slim_scoring_loss01_norm0(float.(λ), X, y, 5e-5) ==
          slim_scoring_loss01_norm0(       λ,  X, y, 5e-5)

end

@testset "column_name_and_index" begin
    df = DataFrame(Cl = Int[1,2], F1 = Float64[1.3, 4.5], F2 = Int[1, 3])
    @test column_name_and_index(df, 1) == (:Cl, 1)
    @test column_name_and_index(df, 2) == (:F1, 2)
    @test column_name_and_index(df, 3) == (:F2, 3)

    @test column_name_and_index(df, :Cl) == (:Cl, 1)
    @test column_name_and_index(df, :F2) == (:F2, 3)
    @test_throws AssertionError column_name_and_index(df, :MyClass)

    @test column_name_and_index(df, "Cl") == (:Cl, 1)
    @test column_name_and_index(df, "F2") == (:F2, 3)
    @test_throws AssertionError column_name_and_index(df, "MyF1")

    @test column_name_and_index(df, "Class"[1:2]) == (:Cl, 1)
    @test_throws AssertionError column_name_and_index(df, "Class"[1:3])
end

@testset "SLIMScoringSystem" begin
end

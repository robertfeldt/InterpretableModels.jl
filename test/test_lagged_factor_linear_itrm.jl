using InterpretableModels: getfactor, parambounds, getintercept, featurename
using InterpretableModels: setfeaturenames!, modelasstring, setparams!, hasparams
using InterpretableModels: minlag, actualminlag, maxlag, actualmaxlag, params, horizon
using InterpretableModels: predict, predict!, fit!
using DataFrames

@testset "LaggedFactorLM" begin

lm = LaggedFactorLITRM(2, 4;
        minlag = 3, maxlag = 8,
        hasintercept = true, mincoef = -10.0, maxcoef = 97.5)

@test !isclassificationmodel(lm)
@test  isregressionmodel(lm)
@test  istimeseriesmodel(lm)
@test !hasparams(lm)
@test minlag(lm) == 3
@test actualminlag(lm) == 3
@test horizon(lm) == 3
@test maxlag(lm) == 8
@test actualmaxlag(lm) == 8

setparams!(lm, [1.0, 2.24, 5.5, 4.1, 1.2, 8.0, 0.5])
@test hasparams(lm)
@test minlag(lm) == 3
@test actualminlag(lm) == 5 # Since 5.5 < 8.0
@test actualmaxlag(lm) == 8 # Since 8.0 > 5.5
@test horizon(lm) == 5
@test params(lm) == [1.0, 2.24, 5.5, 4.1, 1.2, 8.0, 0.5]

c, f, l = getfactor(lm, 1)
@test c == 1.0
@test isa(f, Int)
@test f == 2
@test isa(l, Int)
@test l == 5

c, f, l = getfactor(lm, 2)
@test c == 4.1
@test isa(f, Int)
@test f == 1
@test isa(l, Int)
@test l == 8

@test getintercept(lm) == 0.5

@testset "parambounds" begin
    bs = parambounds(lm)
    @test length(bs) == (3*2+1) # 3 values per factor and 1 for intercept
    for fi in 1:2
        o = (fi-1)*3
        @test bs[1+o] == (-10.0, 97.5)
        @test bs[2+o][1] == 1.0  # feature 1 is lowest possible feature number
        @test isapprox(bs[2+o][2], 1.0+4.0, atol=1e-4)  # Should almost be 1+nfeatures
        @test bs[3+o][1] == 3.0                         # minlag
        @test isapprox(bs[3+o][2], 1.0+8.0, atol=1e-4)  # maxlag
    end
    @test bs[end] == (-10.0, 97.5)
end # @testset "parambounds" begin

@testset "featurenames and setfeaturenames!" begin
    @test featurename(lm, 1) == "feature_1"
    @test featurename(lm, 2) == "feature_2"
    setfeaturenames!(lm, ["a", "bfg"])
    @test featurename(lm, 1) == "a"
    @test featurename(lm, 2) == "bfg"
end # @testset "featurenames and setfeaturenames!" begin

@testset "modelasstring" begin
    setparams!(lm, [1.2, 1.6, 5.8, 6.987, 2.3, 7.3, 0.762])
    @test modelasstring(lm) == "0.76 + 1.2 * a[T-5] + 6.99 * bfg[T-7]"
    df = DataFrame(MyFactor1 = Int[], AnotherFac = Float64[])
    @test modelasstring(lm, df) == "0.76 + 1.2 * MyFactor1[T-5] + 6.99 * AnotherFac[T-7]"
end

@testset "predict, predict! and fit!" begin
    lm = LaggedFactorLITRM(2, 3;
            minlag = 1, maxlag = 3,
            hasintercept = true, mincoef = -10.0, maxcoef = 97.5)

    df = DataFrame(
        FacA = Int[    1,    2,   3,    4,    5],
        Fac2 = Float64[3.14, 5.4, 9.8, 10.0, 25.0],
        Fac3 = Float64[4.00, 6.0, 8.0, 10.0, 12.0])

    X = Array{Float64}(df)

    # Since we have a lag range from 1-3 we can only get two predictions
    # from this model, i.e. for rows 4 and 5.
    # True expression: y = 1.34 .+ X[T-1, 1] .* (-2.5) .+ X[T-2, 3] .* 4.0
    y = [Inf, Inf, Inf,
            1.34 + (-2.5)*X[4-1, 1] + 4.0*X[4-2, 3],
            1.34 + (-2.5)*X[5-1, 1] + 4.0*X[5-2, 3]]
    trueparams = [-2.5, 1.0, 1.0, 4.0, 3.0, 2.0, 1.34]
    yhat = predict(lm, X, trueparams)
    @test length(yhat) == 2
    @test yhat[1] == y[4]
    @test yhat[2] == y[5]

    yhatpreds = zeros(Float64, 2)
    predict!(lm, yhatpreds, X, trueparams)
    @test yhatpreds[1] == y[4]
    @test yhatpreds[2] == y[5]

    # If we don't have the true params we can fit the model to data to recover them.
    dummyparams = [3.5, 1.5, 1.5, 2.1, 2.0, 1.7, 1.34]
    setparams!(lm, dummyparams) # Set a random param vector to ensure a better one is set after fitting
    fit!(lm, X, y)
    @test length(params(lm)) == (1+2*3)
    @test params(lm) != dummyparams

    # Since the problem is underconstrained we cannot expect to find the "true"
    # params since there are many that works...
    yhat = predict(lm, X)
    @test isapprox(yhat[1], y[4], atol=1e-2)
    @test isapprox(yhat[2], y[5], atol=1e-2)
end

end # @testset "LaggedFactorLM" begin

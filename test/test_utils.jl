using InterpretableModels: bbo_floor_int_searchrange, nnzeroasint, sign_predict
using InterpretableModels: sign_accuracy, sign_misclassification_rate, aspct
using InterpretableModels: as_sign_labels, nonzeroindices, uptobutexcluding

@testset "bbo_floor_int_searchrange" begin
    lb, ub = bbo_floor_int_searchrange(5)
    @test floor(Int, lb) == -5
    @test floor(Int, ub) ==  5

    lb, ub = bbo_floor_int_searchrange(10)
    @test floor(Int, lb) == -10
    @test floor(Int, ub) ==  10
end

@testset "nnzeroasint" begin
    @test nnzeroasint(Int[]) == 0
    @test nnzeroasint([1]) == 1
    @test nnzeroasint([0, 1]) == 1
    @test nnzeroasint([2, 0]) == 1
    @test nnzeroasint([3, 3]) == 2
    @test nnzeroasint([4, 0, 5]) == 2
end

@testset "sign_predict" begin
    @test sign_predict([1, 2], [1 -1;-1 0]) == Int[-1, -1]
    @test sign_predict([1.11, 2.456], [1 -1;-1 0]) == Int[-1, -1]

    @test sign_predict([1, 0], [1 -1;-1 0]) == Int[ 1, -1]
    @test sign_predict([1.45, 0.13], [1 -1;-1 0]) == Int[ 1, -1]
end

@testset "sign_accuracy and sign_misclassification_rate" begin
    yhat = y = [1, -1, 1]
    @test sign_accuracy(yhat, y) == 1.0
    @test sign_misclassification_rate(yhat, y) == 0.0
end

@testset "aspct" begin
    @test aspct(0.50) == 50.0
    @test aspct(0.334) == 33.4
end

@testset "as_sign_labels" begin
    @test as_sign_labels([1, -1]) == [1, -1]
    @test as_sign_labels([2, -3]) == [1, -1]
    @test as_sign_labels([-2,  0]) == [-1, -1]
    @test as_sign_labels([-10, 4, 0, 2]) == [-1, 1, -1, 1]

    @test as_sign_labels([-7.6, 3, 0.0]) == [-1, 1, -1]

    @test as_sign_labels([:a, :b, :a]) == [1, -1, 1]
    @test as_sign_labels([:a, :a]) == [1, 1]

    @test as_sign_labels(["a"]) == [1]
    @test as_sign_labels(["bt", "a"]) == [-1, 1]
end

@testset "nonzeroindices" begin
    @test nonzeroindices([1, 0]) == Int[1]
    @test nonzeroindices([1.2, -3]) == Int[1, 2]
    @test nonzeroindices(Float64[]) == Int[]
    @test nonzeroindices(Int[]) == Int[]
    @test nonzeroindices([0]) == Int[]

    @test nonzeroindices([-3, 1.2], false) == Int[1, 2] # Given in same order as stated
    @test nonzeroindices([-3, 1.2], true) == Int[2, 1]  # 2nd one comes before since it has higher value

    @test nonzeroindices([4, 0, -3, 1.2], false) == Int[1, 3, 4]
    @test nonzeroindices([4, 0, -3, 1.2], true) == Int[1, 4, 3]
end

@testset "uptobutexcluding" begin
    @test uptobutexcluding(5, []) == Int[1, 2, 3, 4, 5]
    @test uptobutexcluding(4, [1]) == Int[2, 3, 4]
    @test uptobutexcluding(6, [3, 1]) == Int[2, 4, 5, 6]
    @test uptobutexcluding(3, [3, 1]) == Int[2]

    @test uptobutexcluding(1, [2]) == Int[1]
    @test uptobutexcluding(1, [1]) == Int[]
    @test uptobutexcluding(0, []) == Int[]

    @test uptobutexcluding(3, ["3"]) == Int[1, 2, 3]
end

using Test
begin
 # Sample parameters
 a = 150     # Number of successes
 b = 500     # Number of trials
 c = 0.3     # Hypothesized probability
 
 # 1. Two-sided test
 @testset "Two-sided BinomialTest" begin
     test = BinomialTest(a, b, c)
     @test isapprox(pvalue(test), 1, atol=1e-3) 
 end
 
 # 2. One-sided test (greater)
 @testset "One-sided BinomialTest (greater)" begin
     test = BinomialTest(a, b, c)
     @test isapprox(pvalue(test, tail= :right), 0.5168, atol=1e-3) 
 end
 
 # 3. One-sided test (less)
 @testset "One-sided BinomialTest (less)" begin
     test = BinomialTest(a, b, c)
     @test isapprox(pvalue(test, tail=:left), .52204, atol=1e-3)
 end
 
end

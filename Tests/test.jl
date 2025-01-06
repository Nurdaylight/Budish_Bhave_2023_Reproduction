using Test, HypothesisTests
using DataFrames
using Statistics

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



 begin
 # Sample DataFrame
 df = DataFrame(
     customer_id = [1, 1, 2, 2, 3],
     profit = [100, -50, 200, -300, 150],
     overbid_perc = [25, 50, 100, 75, 30],
     tm_pro = [1, 0, 1, 1, 0]
 )
 
 # 1. Test Data Filtering
 @testset "Data Filtering Test" begin
    filtered_df = filter(:profit => x -> x > 0, df)
    @test nrow(filtered_df) == 3  
 end
 
 # 2. Test Bootstrapped Mean
 @testset "Bootstrapped Mean Test" begin
     mean_profit = mean(df.profit)
     @test isapprox(mean_profit, 20.0, atol=1e-3) 
 end
 
 # 3. Test Conditional Variable Generation
 @testset "Conditional Generation Test" begin
     df.overbid_25 = df.overbid_perc .>= 25
     df.overbid_50 = df.overbid_perc .>= 50
     df.overbid_100 = df.overbid_perc .>= 100
 
     @test sum(df.overbid_25) == 5
     @test sum(df.overbid_50) == 3
     @test sum(df.overbid_100) == 1
 end
 
 # 4. Test Customer Segmentation
 @testset "Customer Segmentation Test" begin
     grouped_df = groupby(df, :customer_id)
     first_bid = combine(grouped_df, :profit => first => :first_profit)
     last_bid = combine(grouped_df, :profit => last => :last_profit)
 
     @test first_bid.first_profit[1] == 100
     @test last_bid.last_profit[1] == -50
 end
 
 # 5. Test Cluster-Level Aggregation
 @testset "Cluster Aggregation Test" begin
     grouped_profit = combine(groupby(df, :customer_id), :profit => mean => :mean_profit)
     @test grouped_profit.mean_profit[1] == 25.0
     @test grouped_profit.mean_profit[2] == -50.0
     @test grouped_profit.mean_profit[3] == 150.0
 end
end
end

module BB23

using Pkg
Pkg.add(["StatFiles", "DataFrames", "Statistics", "GLM", "Plots", "StatsBase",  "OnlineStats",
            "CovarianceMatrices" , "RDatasets", "FixedEffectModels",  "Random", "Bootstrap", "Pipe", "Parameters", "LinearAlgebra", "KernelDensity"])
using StatFiles, DataFrames, Statistics, GLM,  Plots,   OnlineStats, StatsBase, OnlineStats, 
        CovarianceMatrices, RDatasets, FixedEffectModels, Random, Bootstrap, Pipe, Parameters, LinearAlgebra, KernelDensity

pwd()
cd("D:\\Downloads\\repl_ticketmaster\\Misha Data\\Data")
pwd()


begin
	df = DataFrame(load("Matched_data/TM_auctions.dta"))
end


println("How many artists are in the TM data? $(length(unique(df.artist)))")

println("How many concerts are in the TM data? $(size(unique(df[:, [:artist,:event_date]]))[1])")

begin
	df2 = DataFrame(load("Additional_analysis_data/ebay_tm_join_csr.dta"))
end

begin
	plot(df2.tm_ppt_csr, df2.ebay_net_ppt_csr, seriestype=:scatter, label = "eBay secondary-market value")
	
	point1 = (x1 = maximum(df2.ebay_net_ppt_csr), y1 = maximum(df2.ebay_net_ppt_csr))
	point2 = (x2 = 0, y2 = 0)
	x_vals = [point1.x1, point2.x2]
	y_vals = [point1.y1, point2.y2]
	
	plot!(x_vals, y_vals,
		label="unit-slope line",
		xlabel="Ticketmaster primary-market auction price",
		ylabel="eBay secondary-market value")
end



begin
	plot(df2.tm_face_value_csr, df2.ebay_net_ppt_csr,
		seriestype=:scatter, label = "eBay secondary-market value")
	
	plot!(x_vals, y_vals,
		label="unit-slope line",
		xlabel="Ticketmaster face value",
		ylabel="eBay secondary-market value")
end

begin
	df3 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
	mean_prof = mean(df3.profit)
	mean_ppt = mean(df3.tm_ppt)
	
	println("Average profit $(mean_prof) ")
	println("Average TM auction price $(mean_ppt)")
	println("Average profit/average TM auction price is $(100*mean_prof/mean_ppt)")

	mean_profit_counterfact = mean(df3.ebay_net_ppt_csr - df3.tm_face_value)
	println("Average profit secondary market is $(mean_profit_counterfact)")
	
	mean_face_value = mean(df3.tm_face_value)
	println("Average TM face value price is $mean_face_value")
	
	println("The resale profit/average face value price is $(100*mean_profit_counterfact/mean_face_value)")

	df3.ebay_net_ppt_csr = Float64.(df3.ebay_net_ppt_csr)
	df3.tm_face_value = Float64.(df3.tm_face_value)
	model = lm(@formula(ebay_net_ppt_csr ~ tm_face_value), df3)
	println("R^2 of regression of eBay value on TM face value is $(r2(model))")

	df3.tm_ppt = Float64.(df3.tm_ppt)
	model = lm(@formula(ebay_net_ppt_csr ~ tm_ppt), df3)
	println("R^2 of regression of eBay value on TM auction price is $(r2(model))")
	
end


begin
	df4 =  DataFrame(load("Matched_data/TM_auctions.dta"))

	grouped = groupby(df4, :customer_id)
	means = combine(grouped, :tm_pro => mean)
	println(" % of customers in TM data that are experienced bidders $(mean(means.tm_pro_mean))")

	println(" % of transactions in TM data that are done by experienced bidders $(mean(df4.tm_pro))")

	df5 =  DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
	grouped_ = groupby(df5, :tm_pro)
	means_ = combine(grouped_, :profit => mean)
	println(" Profits of experienced bidders are $(means_.profit_mean[2])")
    df5.profit_counterfact = df5.ebay_net_ppt_csr .- df5.tm_face_value
    println("Counterfactual profits under fixed price sales, Mean: $(mean(df5.profit_counterfact))")
    println("Standard Deviation: $(std(df5.profit_counterfact))")
end


# Section 2
begin
	df6 =  DataFrame(load("Matched_data/TM_auctions.dta"))
    df6.tot_fees = df6.facilityfee  .+ df6.tmconveniencefee
    mean(skipmissing(df6[(df6.artist .== "artist14") .& 
                                 (df6.month .== 7) .& (df6.day .==29) .& (df6.year .==2007), :].tot_fees))

end


# Section 4.1

begin
	df7 = DataFrame(load("Matched_data/TM_auctions.dta"))
    vscodedisplay(df7)
    #count of concerts in TM
    df7.concert_counter = groupby(df7, [:artist, :event_date]) |> groupindices
    num_tm_conc=length(unique(df7.concert_counter))
    num_tm_trans = nrow(df7)
    #count of auctions in TM
    df7.auction_counter = groupby(df7, [:artist, :event_date, :auction_id]) |> groupindices
    length(unique(df7.auction_counter))  
    
end

#Share of Canada observations in the TM data Footnote19

begin
	df8 = DataFrame(load("Additional_analysis_data/TM_auctions_Canada.dta"))
    mean(df8.Canada_obs)    
end

#Date range of concerts in primary market data ??? maybe secondary
begin
    df7.date = df7.year .* 10000 .+ df7.month .* 100 .+ df7.day
    minimum(df7.date)
    maximum(df7.date)
    #FIXME - need to check the date range of the Primary market data

    #Number of transactions
    nrow(df7)
    mean(df7.ticket_quantity)
    std(df7.ticket_quantity)
    var(df7.ticket_quantity)
    o = fit!(Moments(),df7.ticket_quantity)
    skewness(o)
    StatsBase.kurtosis(o) #FIXME kurtosis values are not matching
    quantile(df7.ticket_quantity, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
end

#Section 4.2


begin
	df9 = DataFrame(load("Matched_data/ebay_auctions.dta"))
    #there is just a single value
    mean(Int.(df9.num_obs_full_dataset))
    vscodedisplay(df9)   
end


#Footnote 22 and the 4% figure
#Average eBay sale price per ticket (including fees)

begin
	df10 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
    mean(df10.ebay_ppt_csr)
    mean(df10.ebay_fee_pct_csr)
end

#What % of data are multi-object auctions or single-ticket auctions?

begin
	df11 = DataFrame(load("Additional_analysis_data/ebay_auctions_one_ticket_dutch.dta"))
    count(row -> row.dutch == 1 || row.one_ticket == 1, eachrow(df11))/nrow(df11)
    vscodedisplay(df11)
end

#SECTION 4.3
#Table 1: Comparison of raw and matched data
begin
	df7.csr_counter = groupby(df7, [:artist, :event_date, :section, :row]) |> groupindices
    num_tm_csr=length(unique(df7.csr_counter))
     
end

begin
	df10.concert_counter= groupby(df10, [:artist, :event_date]) |> groupindices
    num_matched_conc= length(unique(df10.concert_counter))
    df10.csr_counter = groupby(df10, [:artist, :event_date, :section, :row]) |> groupindices
    num_matched_csr =length(unique(df10.csr_counter))
    num_matched_trans= nrow(df10)
    share_matched_tm = 100*num_matched_conc/num_tm_conc
    share_matched_csr=100*num_matched_csr/num_tm_csr
    share_matched_trnsc=100*num_matched_trans/num_tm_trans
end

#Summary statistics for TM transactions per CSR in matched dataset

begin
    df10.tm_trans_csr = combine(groupby(df10, [:artist, :event_date, :section, :row]), eachindex)[:, 5]
    per_csr_chars = combine(groupby(df10, [:artist, :event_date, :section, :row]), nrow)[:, 5]
    mean(per_csr_chars)
    std(per_csr_chars)
    quantile(per_csr_chars, [0.25,0.75])
end

#Summary statistics for eBay transactions per CSR in matched dataset

begin
    df14 =  DataFrame(load("Additional_analysis_data/ebay_tm_join_ebay.dta"))
    ebay_trans_csr_chars = combine(groupby(df14, [:artist, :event_date, :section, :row]), nrow)[:, 5]
    #How many eBay transactions are in the matched data?
    mean(ebay_trans_csr_chars)
    std(ebay_trans_csr_chars)
    quantile(ebay_trans_csr_chars, [0.25,0.75])
    nrow(df14)

end






#Section 5

begin
	df15 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
	mean_prof = mean(df15.profit)
	mean_tm_ppt = mean(df15.tm_ppt)
	println("Mean profit: $mean_prof")
	println("Average TM auction price: $mean_tm_ppt")
	println("Average profit/average TM auction price: $(100*mean_prof/mean_tm_ppt)")
end

begin
	Random.seed!(1)
	
	grouped_data = groupby(df15, [:artist, :event_date])
	bootstrap_clust = []
	n_iter = 2000
	
	for _ in 1:n_iter
		df_b = DataFrame([])
		for i in sample(1:length(grouped_data), length(grouped_data); replace = true, ordered = false)
			df_b = vcat(df_b, grouped_data[i])
		end
		push!(bootstrap_clust, mean(df_b.profit))
	end

	print(
		"""
		Clustered bootstrap: $(mean(bootstrap_clust)),
		CI: $(quantile(bootstrap_clust, [0.025, 0.975]))
		"""
	)
    histogram(bootstrap_clust)
end

begin
	n_boot = 2000
	bs1 = bootstrap(mean, df15.profit, BasicSampling(n_boot))
	## balanced bootstrap
  	bs2 = bootstrap(mean, df15.profit, BalancedSampling(n_boot))
	
	## calculate 95% confidence intervals
	cil = 0.95
	
	## basic CI
	bci1 = confint(bs1, BasicConfInt(cil))
	## percentile CI
  	bci2 = confint(bs1, PercentileConfInt(cil));
	## BCa CI
	bci3 = confint(bs1, BCaConfInt(cil));
	## Normal CI
	bci4 = confint(bs1, NormalConfInt(cil));
	print(
		"""Different types of CI for basic bootstrap:
			$(bci1...),
			$(bci2...),
			$(bci3...),
			$(bci4...)
		"""
	)

	bci1_2 = confint(bs2, BasicConfInt(cil))
  	bci2_2 = confint(bs2, PercentileConfInt(cil));
	bci3_2 = confint(bs2, BCaConfInt(cil));
	bci4_2 = confint(bs2, NormalConfInt(cil));

	print(
		"""Different types of CI for basic bootstrap:
			$(bci1_2...),
			$(bci2_2...),
			$(bci3_2...),
			$(bci4_2...)
		"""
	)
    frac_not_outlier = count(abs.(df15.profit) .< 500)
    df15.weiii .= frac_not_outlier
    histogram(bins=80,  normalize=true,
        df15.profit[abs.(df15.profit) .< 500],
        xlabel="eBay secondary-market value minus Ticketmaster primary-market auction price", 
        ylabel="fraction of matched obs. with this resale profit level",
        title="Figure 3 Reproduction",
        guidefont=font(9) )
    
    yticks!(0:0.850078386514147e-3:7e-3, string.( 0:0.02:0.16))
end

   #section 6

begin
	df16 = DataFrame(load("Matched_data/TM_auctions.dta"))
	num_obs= nrow(df16)
	won_one=  sum(df16.num_trans_cust .== 1)/num_obs
    win2_9 = length(df16.num_trans_cust[df16.num_trans_cust .<= 9])/num_obs - won_one
    win_tenplus = length(df16.num_trans_cust[df16.num_trans_cust .>= 10])/num_obs
end

#alternative defenition of experienced

begin
    

    df16 = DataFrame(load("Matched_data/TM_auctions.dta"))
	mean(combine(groupby(df17, :customer_id), :tm_pro2 => mean)[:,2])
    mean(Int.(df16.tm_pro2))
    mean(df16.cust_unique[df17.cust_unique .==1])

end
begin
    df17 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
    experienced_profit =  Float64.(df17.profit[df17.tm_pro .== 1])
    inexperienced_profit =  Float64.(df17.profit[df17.tm_pro .== 0])
   
    println("Profits of experienced bidders: $(mean(experienced_profit))")
    println("Profits of inexperienced bidders: $(mean(inexperienced_profit))")
    
    # # Compute kernel density estimates
    kde_exp = kde(experienced_profit, boundary = (-500, 500), npoints = 3000)
    kde_inexp = kde(inexperienced_profit, boundary = (-500, 500), npoints = 3000)
    
    plot(
        kde_exp.x, kde_exp.density,
        label = "Experienced Bidders",
        xlabel = "eBay secondary-market value minus Ticketmaster primary-market auction price",
        ylabel = "Kernel Density",
        linewidth = 2,
        guidefont=font(9)
    )
    
    plot!(
        kde_inexp.x, kde_inexp.density,
        linestyle = :dash,
        label = "Inexperienced Bidders",
        linewidth = 2
    )
   
   

end



#SECTION 6.2
#Overbidding analysis
begin
    df20 = DataFrame(load("Matched_data/TM_auctions.dta"))
    num_obs_perc= df20
    total_obs = count(row -> row.guideline_group == 1, eachrow(df20))
    for i in [25,50,100]
        obs = count(row -> row.overbid_perc  .>=i && row.guideline_group == 1, eachrow(df20))/total_obs
        println("Percentage of observations with overbidding percentage greater than $i%: $obs")
    end
    #*What % of winning bids are within $10 of the next-highest winning bid?
    for x in [10,50]
        within = count(x -> x == true, skipmissing(df20.increment .<= x))/nrow(df20)
        println("Percentage of winning bids within $x USD of the next-highest winning bid: $within")
    end
end

#Table 3 Work
begin
    df21 = DataFrame(load("Additional_analysis_data/TM_losers.dta")) 
    df_over = vcat(df20, df21, cols = :union)
    sort!(df_over, [:customer_id, :TM_aucend_date])
    
    df_over.first_bid .= 0
    df_over.last_bid .= 0


        
end


end # module BB23

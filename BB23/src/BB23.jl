

using Pkg
Pkg.add(["StatFiles", "DataFrames", "Statistics", "GLM", "Plots", "StatsBase",  "OnlineStats",
            "CovarianceMatrices" , "RDatasets", "FixedEffectModels",  "Random", "Bootstrap", "Pipe", "Parameters", "LinearAlgebra", "KernelDensity", "Distributions"])

module BB23
using StatFiles, DataFrames, Statistics, GLM,  Plots,   OnlineStats, StatsBase, OnlineStats, 
        CovarianceMatrices, RDatasets, FixedEffectModels, Random, Bootstrap, Pipe, Parameters, LinearAlgebra, KernelDensity, Distributions

pwd()
cd("D:\\Downloads\\repl_ticketmaster\\Misha Data\\Data")
pwd()

export run

function run() 
    begin
        df = DataFrame(load("Matched_data/TM_auctions.dta"))
        

        println("How many artists are in the TM data? $(length(unique(df.artist)))")

        println("How many concerts are in the TM data? $(size(unique(df[:, [:artist,:event_date]]))[1])")
    end


    begin
        df2 = DataFrame(load("Additional_analysis_data/ebay_tm_join_csr.dta"))
   
        plot(df2.tm_ppt_csr, df2.ebay_net_ppt_csr, seriestype=:scatter, label = "eBay secondary-market value")
        
        point1 = (x1 = maximum(df2.ebay_net_ppt_csr), y1 = maximum(df2.ebay_net_ppt_csr))
        point2 = (x2 = 0, y2 = 0)
        x_vals = [point1.x1, point2.x2]
        y_vals = [point1.y1, point2.y2]
        
        plot!(x_vals, y_vals,
            label="unit-slope line",
            xlabel="Ticketmaster primary-market auction price",
            ylabel="eBay secondary-market value")
            savefig("./Output/figure_1_a.png")
        plot(df2.tm_face_value_csr, df2.ebay_net_ppt_csr,
            seriestype=:scatter, label = "eBay secondary-market value")
        
        plot!(x_vals, y_vals,
            label="unit-slope line",
            xlabel="Ticketmaster face value",
            ylabel="eBay secondary-market value")
            savefig("./Output/figure_1_b.png")
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
        fees = mean(skipmissing(df6[(df6.artist .== "artist14") .& 
                                    (df6.month .== 7) .& (df6.day .==29) .& (df6.year .==2007), :].tot_fees))
        println(" average TM and venue fees $fees")

    end


    # Section 4.1

    begin
        df7 = DataFrame(load("Matched_data/TM_auctions.dta"))
        #count of concerts in TM
        df7.concert_counter = groupby(df7, [:artist, :event_date]) |> groupindices
        num_tm_conc=length(unique(df7.concert_counter))
        num_tm_trans = nrow(df7)
        #count of auctions in TM
        df7.auction_counter = groupby(df7, [:artist, :event_date, :auction_id]) |> groupindices
        auctions = length(unique(df7.auction_counter))  

        println(" TM concerts number $num_tm_conc")
        println(" TM winning bids number $num_tm_trans")
        println(" TM auction number $auctions")
        
    end

    #Share of Canada observations in the TM data Footnote19

    begin
        df8 = DataFrame(load("Additional_analysis_data/TM_auctions_Canada.dta"))
        can=mean(df8.Canada_obs)  
        println(println(" Share of Canadian observations $can"))  
    end

    #Date range of concerts in primary market data ??? maybe secondary
    begin
        df7.date = df7.year .* 10000 .+ df7.month .* 100 .+ df7.day
        mindat=minimum(df7.date)
        maxdat=maximum(df7.date)
        println(" Date range of concerts in primary market data; Earliest event date $mindat")
        println(" Latest event date $maxdat ")
    
        #Number of transactions
        ntrans=nrow(df7)
        println(" Number of transactions in TM data  $ntrans")

        mean_tickq = mean(df7.ticket_quantity)
        std_tickq = std(df7.ticket_quantity)
        var_tickq = var(df7.ticket_quantity)
        o = fit!(Moments(), df7.ticket_quantity)
        skew_tickq = skewness(o)
        kurt_tickq = kurtosis(o) # Kurtosis is based on Excess Kurtosis
        quantiles = quantile(df7.ticket_quantity, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])

        # Printing the summary
        println("Summary of ticket quantity:")
        println("  Mean: $mean_tickq")
        println("  Standard Deviation: $std_tickq")
        println("  Variance: $var_tickq")
        println("  Skewness: $skew_tickq")
        println("  Kurtosis: $kurt_tickq")
        println("  Quantiles:")
        println("    1%: $(quantiles[1]), 5%: $(quantiles[2]), 10%: $(quantiles[3])")
        println("    25%: $(quantiles[4]), 50% (Median): $(quantiles[5])")
        println("    75%: $(quantiles[6]), 90%: $(quantiles[7]), 95%: $(quantiles[8]), 99%: $(quantiles[9])")
    end

    #Section 4.2


    begin
        df9 = DataFrame(load("Matched_data/ebay_auctions.dta"))
        #there is just a single value        
        println("How many files are in the raw eBay data? $(mean(Int.(df9.num_obs_full_dataset)))")
    end


    #Footnote 22 and the 4% figure
    #Average eBay sale price per ticket (including fees)

    begin
        df10 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
        mean(df10.ebay_ppt_csr)
        mean(df10.ebay_fee_pct_csr)
        println("Average eBay sale price per ticket (including fees) $(mean(df10.ebay_ppt_csr))")
        println("Average eBay fee percentage $(mean(df10.ebay_fee_pct_csr))")
    end

    #What % of data are multi-object auctions or single-ticket auctions?

    begin
        df11 = DataFrame(load("Additional_analysis_data/ebay_auctions_one_ticket_dutch.dta"))
        count(row -> row.dutch == 1 || row.one_ticket == 1, eachrow(df11))/nrow(df11)
   
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

    begin #FIXME adjust iteration number to proper one
        df15 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
        mean_prof = mean(df15.profit)
        mean_tm_ppt = mean(df15.tm_ppt)
        println("Mean profit: $mean_prof")
        println("Average TM auction price: $mean_tm_ppt")
        println("Average profit/average TM auction price: $(100*mean_prof/mean_tm_ppt)")



        Random.seed!(1)
        
        grouped_data = groupby(df15, [:artist, :event_date])
        bootstrap_clust = []
        n_iter = 2
        
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
        savefig("./Output/bootstrap.png")

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
        savefig("./Output/figure_3.png")
    end

    #section 6

    begin
        df16 = DataFrame(load("Matched_data/TM_auctions.dta"))
        num_obs= nrow(df16)
        won_one=  sum(df16.num_trans_cust .== 1)/num_obs
        win2_9 = length(df16.num_trans_cust[df16.num_trans_cust .<= 9])/num_obs - won_one
        win_tenplus = length(df16.num_trans_cust[df16.num_trans_cust .>= 10])/num_obs
        print(" % of transactions in matched data are done by bidders who win 1 TM auctions overall: $won_one
        % of transactions in matched data are done by bidders who win 2-9 TM auctions overall: $win2_9
        % of transactions in matched data are done by bidders who win 10 or more TM auctions overall: $win_tenplus")

    end

    #alternative defenition of experienced

    begin      

        df16 = DataFrame(load("Matched_data/TM_auctions.dta"))
        altern_exp_ = mean(combine(groupby(df16, :customer_id), :tm_pro2 => mean)[:,2])
        altern_exp_trans = mean(Int.(df16.tm_pro2))

        print("  % of customers in matched data are experienced bidders as per alternative definition: $altern_exp_
        % of transactions in matched data are done by experienced bidders as per alternative definition of experience: $altern_exp_trans
        ")
        

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
        savefig("./Output/figure_4.png")
    
    

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
        # Set seed
        Random.seed!(100)
        df2 = DataFrame(load("Additional_analysis_data/ebay_tm_join.dta"))
        # Define the function to calculate mean difference
        function pro_fan_diff(df)
            pro_mean = mean(skipmissing(df.profit[df.tm_pro .== 1]))
            fan_mean = mean(skipmissing(df.profit[df.tm_pro .== 0]))
            return pro_mean - fan_mean
        end
        
        grouped_data = groupby(df2, [:artist, :event_date])
        bootstrap_clust = []
        n_iter = 200

        
        for _ in 1:n_iter
            df_b = DataFrame([])
            for i in sample(1:length(grouped_data), length(grouped_data); replace = true, ordered = false)
                df_b = vcat(df_b, grouped_data[i])
            end
            push!(bootstrap_clust, pro_fan_diff(df_b))
        end
    
        print(
            """
            Clustered bootstrap: $(mean(bootstrap_clust)),
            CI(99%): $(quantile(bootstrap_clust, [0.005, 0.995]))
            """
        )
    end

    begin
        df2.tm_fan = 1 .- df2.tm_pro
        # Number of experienced bidders
        num_pro = sum(df2.tm_pro .== 1)
        
        # Number of inexperienced bidders
        num_fan = nrow(df2) - num_pro
        
        println("Number of experienced bidders: ", num_pro)
        println("Number of inexperienced bidders: ", num_fan)

        df2.concert = groupindices(grouped_data)
        df2.csr = groupindices(groupby(df2, [:artist, :event_date, :section, :row]))
        df2.obs = 1:nrow(df2)
    
    end

    begin
        bid_shares_csr = combine(groupby(df2, [:csr]), :tm_pro => sum => :tm_pro_total, :tm_fan => sum => :tm_fan_total, :profit => mean => :csr_mean_profit)
        bid_shares_csr.share_pro_csr = bid_shares_csr.tm_pro_total ./ sum(bid_shares_csr.tm_pro_total)
        bid_shares_csr.share_fan_csr = bid_shares_csr.tm_fan_total ./ sum(bid_shares_csr.tm_fan_total)
    
        bid_shares_csr
    end	

    begin
        println(" $(mean(df2.profit_fv))")
    
        # Create new columns based on the profit conditions
        df2.small_pos_prof = ifelse.(df2.profit .>= 0 .&& df2.profit .<= 100, 1, 0)
        df2.large_neg_prof = ifelse.(df2.profit .< -100, 1, 0)
        
        # Function to calculate the share for each group
        function calc_share(df, var, tm_pro_val)
            mean(skipmissing(df[df.tm_pro .== tm_pro_val, var]))
        end
        
        # Calculate shares for both variables and bidder types
        for var in [:small_pos_prof, :large_neg_prof]
            for tm_pro_val in [0, 1]
                share = calc_share(df2, var, tm_pro_val)
                println("Share for $var and tm_pro == $tm_pro_val: ", round(share, digits=4))
            end
        end
    
    end


    begin
        function pro_fan_prof_diff(df)
            results = Dict()
        
            for var in [:small_pos_prof, :large_neg_prof]
                # Calculate the mean for each group (experienced/inexperienced bidders)
                mean_pro = mean(skipmissing(df[df.tm_pro .== 1, var]))
                mean_fan = mean(skipmissing(df[df.tm_pro .== 0, var]))
        
                # Calculate the difference between experienced and inexperienced bidders
                results["$(var)_diff_pro_fan"] = mean_pro - mean_fan
            end
        
            return (results["small_pos_prof_diff_pro_fan"], results["large_neg_prof_diff_pro_fan"])
        end
    
    
        bootstrap_clust_diff_pos = []
        bootstrap_clust_diff_neg = []
        
        for _ in 1:n_iter
            df_b_new = DataFrame([])
            for i in sample(1:length(grouped_data), length(grouped_data); replace = true, ordered = false)
                df_b_new = vcat(df_b_new, grouped_data[i])
            end
            push!(bootstrap_clust_diff_pos, pro_fan_prof_diff(df_b_new)[1])
            push!(bootstrap_clust_diff_neg, pro_fan_prof_diff(df_b_new)[2])
        end
    
        print(
            """
            Clustered bootstrap pos: $(mean(bootstrap_clust_diff_pos)),
            CI(99%) pos: $(quantile(bootstrap_clust_diff_pos, [0.005, 0.995]))
    
            Clustered bootstrap neg: $(mean(bootstrap_clust_diff_neg)),
            CI(99%) pos: $(quantile(bootstrap_clust_diff_neg, [0.005, 0.995]))
            """
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

    #Table 3
    begin

        df21 = DataFrame(load("Additional_analysis_data/TM_losers.dta")) 
        df21 = vcat(df20, df21, cols = :union)
        
        # df21 = DataFrame(load("Additional_analysis_data/TM_losers.dta"))

        # Sort the DataFrame by customer_id and TM_aucend_date
        sort!(df21, [:customer_id, :TM_aucend_date])
        
        filter!(row -> !ismissing(row.customer_id), df21)
        filter!(row -> !ismissing(row.TM_aucend_date), df21)
        
        df21.first_bid = zeros(nrow(df21)) 
        df21.last_bid = zeros(nrow(df21)) 
        
        df21_gr_custromer = groupby(df21, :customer_id)

        for g in df21_gr_custromer
            g[1,:first_bid] = 1
            g[nrow(g), :last_bid] = 1
        end

        df21 = DataFrame(df21_gr_custromer)

        df21_gr_custromer_tm_date = combine(
        groupby(df21, [:customer_id, :TM_aucend_date]), 
        :first_bid => maximum => :max_first, 
        :last_bid => maximum => :max_last
        )

        df21 = innerjoin(df21, df21_gr_custromer_tm_date, on = [:customer_id, :TM_aucend_date])

        df21.first_bid .= df21.max_first
        df21.last_bid .= df21.max_last

        df21.only_bid = Float64.(df21.first_bid .+ df21.last_bid .== 2)

        df21_filtered = filter(row -> ifelse(!ismissing(row.guideline_group), row.guideline_group, 0) == 1 && !ismissing(row.tm_ppt), df21)

        df21_filtered.tm_fan .= 1 .- df21_filtered.tm_pro

        
        df21_filtered.overbid_25 = df21_filtered.overbid_perc .>= 25
        df21_filtered.overbid_50 = df21_filtered.overbid_perc .>= 50
        df21_filtered.overbid_100 = df21_filtered.overbid_perc .>= 100

        overbid_25_obs = sum(df21_filtered.overbid_25)
        overbid_50_obs = sum(df21_filtered.overbid_50)
        overbid_100_obs = sum(df21_filtered.overbid_100)
        
    end

    begin
        for perc in [25, 50, 100]
        
            var_list = [:tm_pro, :tm_fan, :first_bid, :last_bid, :only_bid]
        
            for var in var_list
        
                prob_var = mean(df21_filtered[:,var])
                
                println(prob_var)
                
                df21_filtered[!, "overbid_$(perc)_in_$(var)"] .= ifelse.(df21_filtered[!, var] .== 1 .&& df21_filtered.overbid_perc .>= perc, 1, 0)
                
                # Count the number of observations where the condition is met
                count_condition = sum((df21_filtered[!, var] .== 1) .&(df21_filtered.overbid_perc .>= perc))
                
                # Calculate the total number of observations that meet the condition
                overbid_perc_obs = sum(df21_filtered.overbid_perc .>= perc)  
            
                # Display the ratio
                println(count_condition / overbid_perc_obs)
    
                # display the binomial test
                println(BinomialTest(count_condition, overbid_perc_obs, prob_var))
            
                bootstrap_ci_level = 99
                
                if var == :last_bid && perc == 100
                    bootstrap_ci_level = 95 
                    
                elseif (var == :first_bid || var == :only_bid) && perc == 100
                    bootstrap_ci_level = 90
                    
                end
            
                bootstrap_ci_level = bootstrap_ci_level / 100
                print(bootstrap_ci_level)
            
                Random.seed!(100)
            
                grouped_data_art_event = groupby(df21_filtered, [:artist, :event_date])
            
                boots_res = bootstrap_clust_func(grouped_data_art_event, mean_overbid, var, perc, 10)
            
                print(
                    """
                    Clustered bootstrap for $var: $(mean(boots_res)),
                    CI(99%) pos: $(quantile(boots_res, [(1 - bootstrap_ci_level) / 2 , 1 - (1 - bootstrap_ci_level) / 2]))
                    """
                )
            end
        end
    end

    begin
        function mean_overbid(df, var, perc)
            
            df_filtered = filter(row -> row[:overbid_perc] >= perc, df)
            return mean(df_filtered[:, var])
        end
    
        function bootstrap_clust_func(grouped_data, func,  var, perc, n_iter)
            
            bootstrap_clust = []
            
            for _ in 1:n_iter
                
                df = DataFrame([])
                sample_ = sample(1:length(grouped_data), length(grouped_data); replace = true, ordered = false)
                
                for i in sample_
                
                    df = vcat(df, grouped_data[i])
                end
                push!(bootstrap_clust, func(df, var, perc))
            end
            return bootstrap_clust
        end
    end




end #end run

end # module BB23

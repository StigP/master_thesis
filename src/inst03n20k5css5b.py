"""
Version 110.

This file generates instances and runs the entire simulation
and optimization for that instance.

Current example run is the stochastic instance: inst03n20k5cds5b.
Instance 03:
It has n=20 flights, k=5 deice servers
Objective: Min Total Cost (c)
Deterministic Optimization (d)
Stochastic Scenario Evaluation (s)
Buffer: Fixed 5 minutes (5b)
Noise: AOBT: 5 min uniform noise. ADIT: 0.25 sigma lognormal.
"""
import time
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt


#Import of own modules:
# Generate Instances:
from utils import generate_flight_schedule, policy_dict,fcfs_managed
# Simulation and Monte Carlo Simulation:
from simulation import FlightSimulation, MonteCarloSimulation
# Optimizer:
from optimization import TSATOptimization


"""01 Config of this Instance Run:"""
# Seed Management
instance_seed = 8 # For instance generation

crn_use_on_test = True #Common Random Numbers method on test
crn_use_on_train = True #Common Random Numbers method on train

crn_seed_test = 2 # Common Random Number seed for actual realizations testing FCFS vs OPT
crn_seed_train = 8 # Common Random Number seed for actual realizations training in optimizer

optim_seed = 8 # For the optimizer stochastics

surrogat = False # If C(x) evaluations can be done by surrogat model f(x), Disabled.

#Buffer optim?
buffer_optim="no" #min or prob, if no set fixed minutes buffer
buff = 5 #minutes start buffer, or if prob: prob (0,1)

name ="inst03n20k5css5b"

short_summary = "Cost optimization. Assumes FCFS is managed. 5 min buffer."
printout = False #Creates a log if True and print to terminal.
plots = True #Creates plots if True

#Run simulation in Deterministic Mode or Stochastic Mode:
sim_mode = "Sto" #"Det" or "Sto". Deterministic or Stochastic

#Monte Carlo Simulation (for mode = "Sto") or Single Run:
montecarlo = True #True or False if MC run.
M_runs_sim = 2000
M_runs_opt = 128 #For the full optimizer runs.
# MC with antithetic variates:
antithetic= True

if not montecarlo:
    M_runs_sim = 1
    antithetic = False

#Decide if TSAT_optimizer should run for TSAT Optim:
TSAT_optimization = True
if not TSAT_optimization:
    optimizer_method = "None"
if TSAT_optimization:

    """Set target for the optimization
    throughput disregards any other costs,
    total cost includes the cost params for 
    throughput loss, delay costs and wait costs"""

    #objective = "throughput" #throughput or total_cost
    
    #objective = "makespan" #Same as throughput but other plots.
    objective = "total_cost"

    """Select Optimizer for TSAT optimization:
        Either:
        DE: Differential Evolution (scipy)
        DA: Dual Annealing (scipy)
        BF: Brute Force
        RS: Random Search
        TS: Tailored -  Greedy with No Local Search
        TSLS: Tailored - with Local Search
        TSSA: Tailored with Simulated Annealing
        TSLSSAM: Tailored with LS and multiple SA
        TSLSSAS: Tailored with LS and single SA
        TSRW: Tailored with Random Walk
        TSGD:Tailored with Greedy Descent
        """
        


    optimizer_method = "Multiple"
    # Define a dictionary of optimizers to run
    optimizers_to_run = {
    "DE": "DE",
    "DA": "DA",
    #"Brute Force": "BF",
    "RS": "RS",
    "TS": "TS",
    "TSLS": "TS_LS",
    "TSSA": "TS_SA",
    "TSLSSAM": "TS_LS_SA",
    "TSLSSAS": "TS_LS_SA_2",
    "TSRW":"TSRW",
    "TSGD":"TSGD"
    }

    """Set if the optimizer should optimize deterministicly
    or stochasticly. If stochastic it need multiple
    Monte Carlo runs, so set M accordingly"""

    opt_mode = "Sto" #"Det" or "Sto"

    """Optimizer terminates after:"""
    max_optimizer_time = 10 #Seconds run per each optimizer

    stop_at_bound = True #Terminate if reach bound, disable on testing.
    if opt_mode == "Sto":
        stop_at_bound = False

"""Set Policy for resequencing flights after deice
see utils.py for details of this function:"""
#Policy_ICE = "RESEQ_ICE" # or "FCFS"
Policy_ICE = "FCFS" # or "FCFS"
max_wait_after_deice = 1.0 #Wait time if RESEQ_ICE
policies = policy_dict(Policy_ICE, max_wait_after_deice)

"""02 Set up the Airport Environment"""
deicing_servers = 5 #Number of deice pads available.
throughput_cost = 1000 # The cost of the loss of 1 minute makespan efficiency iaw Appendix C.1.

"""03 Flight Schedule - Params to create 
a random flt schedule for the simulation.
See utils.py for details of this function"""

num_flights =20
mean_interarrival_times = 0
five_min_interval = False #Sets times in 5 min intervals
#Define start and end time window of the SOBT of flights considered:
SOBT_start = 0 #Scheduled Off Block Time (SOBT)
SOBT_end = int(num_flights*mean_interarrival_times)

#Distribution of SID compass directions:
#North, East, South = 0.4,0.2,0.2 Last instance.
#North, East, South = 0.4,0.2,0.2
North, East, South = 0.9,0,0
West = 1-(North+East+South)
SID_dist = North, East, South, West

#Distribution of acft categories:
Heavy = 0
#Heavy = 0.2
Medium = 1 - Heavy
CAT_dist = Heavy, Medium

#Max,min expected EXOT (taxi out times):
EXOT_min,EXOT_max = 10,10
taxitime_range = EXOT_min,EXOT_max

#Deice service rates - info given at dep clearance request, around 15 min before EOBT:
#Assume for each flight we can estimate an expected ADIT deice time, within this range.
#The actual ADIT deice time is drawn from lognormal dist with this expected as mean.
service_rate_low = 4.0
service_rate_high = 22.0
#Prob that a flight need deice, otherwise bypass deice.
#ice_need = 1.0
ice_need = 0.8

deice_service_rates = service_rate_low,service_rate_high

#https://ansperformance.eu/economics/cba/standard-inputs/chapters/cost_of_delay.html

delay_cost_weight_factor = 1  # Dummy factor to scale cost below

# Parameters for B738 (catM) b_2, b_1 iaw Appendix C.2:
delay_cost_at_gate_params_catM = (2.2589, 9.0455)
delay_cost_taxiing_params_catM = (2.2802, 23.0717)

# Parameters for A332 (catH) b_2, b_1 iaw Appendix C.2:
delay_cost_at_gate_params_catH = (4.0884, 17.8682)
delay_cost_taxiing_params_catH = (4.0699, 54.7981)

# Cost parameters grouped together
cost_params = (
    delay_cost_at_gate_params_catM,
    delay_cost_taxiing_params_catM,
    delay_cost_at_gate_params_catH,
    delay_cost_taxiing_params_catH
)

# Scale with the weight factor
cost_params = tuple(tuple(param * delay_cost_weight_factor for param in params) for params in cost_params)

# ATC management of FCFS queue to reduce waiting at taxiways?
fcfs_management = True #ATC Manages the queue - releases from gate and push and hold.

# Stochastic noise
noise_params = {
    'exot_noise': 0, #In all instances we have assumed EXOT is deterministic.
    'aobt_noise': 5,
    'adit_sigma': 0.25
}

# Create the flight schedule:
flight_schedule = generate_flight_schedule(num_flights, SOBT_start, SOBT_end, SID_dist, CAT_dist, taxitime_range,deice_service_rates,ice_need,cost_params, noise_params,instance_seed,five_min_interval)


if fcfs_management: # FCFS schedule with managed from gate:
    flight_schedule = fcfs_managed(flight_schedule, deicing_servers,buff)


def run_example(flight_schedule,deicing_servers,throughput_cost,crn_use_on_train,crn_use_on_test,crn_seed_train,crn_seed_test,optim_seed,
                sim_mode = "Det",opt_mode ="Det", policies = "FCFS", 
                montecarlo= False, M_runs_sim = 1,M_runs_opt=1, 
                TSAT_optimization = False, optimizer_method = None, 
                printout=False, 
                plots=False, 
                name="no_name", antithetic=False,surrogat=False,buffer_optim=None,buff=0):

    # Example-specific logic
    print(f'Example: {name} is running')
    # Create a DataFrame
    df = pd.DataFrame(flight_schedule)
    # Save DataFrame to CSV
    df.to_csv(f'{name}_flight_schedule.csv', index=False)

    # Print the DataFrame:
    if printout:
        print("The flight schedule is:")
        print(df.to_string(index=False))
       

    """PARAMS for the sim"""
    # Start timing of the simulation run:
    start_time = time.time()

    Deice_serve = [deicing_servers] #Set number of deice servers. Plots for each value.

    """Optimization Policy for resequencing flights:"""
    if policies == "FCFS":
        Policy_ICE = "FCFS"
        max_wait_after_deice = 0
        policies = policy_dict(Policy_ICE, max_wait_after_deice)


    """FCFS Simulation:"""
    max_sim_time = 1000 #To avoid infinity runs.
    #Set CRN seed:
    np.random.seed(crn_seed_test)
    random.seed(crn_seed_test)

    simulation_FCFS = FlightSimulation(flight_schedule, deicing_servers, throughput_cost, policies, printout, sim_mode,max_sim_time)
    simulation_FCFS.run()


    """Monte Carlo Simulations:"""
    M = M_runs_sim
    burn_in = round(M*0.05) #remove from plot the first burn_in values.


    if montecarlo:
        #Set CRN seed:
        np.random.seed(crn_seed_test)
        random.seed(crn_seed_test)

        mc_simulation_FCFS = MonteCarloSimulation(M, flight_schedule, deicing_servers, throughput_cost, policies, printout,sim_mode,max_sim_time, crn_seed_test,antithetic)
        mc_simulation_FCFS.run()
        if printout:
            print(f"Example: {name} MC Simulation {M} runs FCFS Mean Makespan")
            #print(mc_simulation_FCFS.mean_rwy_throughput)
            print(mc_simulation_FCFS.mean_makespan)

            
    """OPTIMISATION based on deterministic flights (no Monte Carlo)"""
    if TSAT_optimization:
        if opt_mode=="Det":
            M_runs_opt = 1 #Only one MC run to save time if deterministic optimization.
        
        """Train optimizers:"""
        #Set CRN seed (also reset per optimizer as attribute):
        np.random.seed(optim_seed)
        random.seed(optim_seed)
        

        optimizer_runs = TSATOptimization(flight_schedule, deicing_servers, throughput_cost, policies,printout,max_sim_time,M_runs_opt,opt_mode,None, objective,max_optimizer_time, stop_at_bound,optim_seed,crn_seed_train,crn_use_on_train, antithetic,surrogat,buffer_optim,buff)
        results, best_optimizer = optimizer_runs.run_optimizers(optimizers_to_run)
        
        best_TSAT = best_optimizer[1]["best_TSAT"]
        best_objective = best_optimizer[1]["optimized_objective"]
        
       
        # End timing
        end_opttime = time.time()
        execution_time_opt = end_opttime - start_time

        if printout:
            print("best Optim method is:", best_optimizer[0])
            print("Optimized TSATs:", best_TSAT)
            print("Optimized Objective Value:", best_objective)
            print("After exec time:",execution_time_opt)
           
    
    """Optimizer Plots"""
    if plots and TSAT_optimization:       
            plot_name_optimizer = f"{name}"
            optimizer_runs.plot_combined_results(plot_name_optimizer)
          
    if TSAT_optimization:
        # Save results to CSV and Latex:
        report_name_optimizer = f"{name}"
        optimizer_runs.save_results_report(report_name_optimizer,short_summary)
        optimizer_runs.save_results_report_pdf(report_name_optimizer,short_summary)

        

    """New simulation with the optimized TSATS, with CRN Test seed"""
    
    if TSAT_optimization:#Init now with the optimized TSATS found with CRN seed:
        # CRN seed:
        np.random.seed(crn_seed_test)
        random.seed(crn_seed_test)
        simulation_OPT = FlightSimulation(flight_schedule, deicing_servers, throughput_cost, policies, printout,sim_mode,max_sim_time)
        simulation_OPT.TSATs = best_TSAT
        
        simulation_OPT.run()
        if printout:
            print("The RWY Interdep Times with FCFS are:")
            print(simulation_FCFS.runway_interdep_times)
            print("The RWY Interdep Times with the OPT TSATS are:")
            print(simulation_OPT.runway_interdep_times)
            print("The RWY Througput with FCFS are:")
            print(simulation_FCFS.runway_throughput[-1])
            print("The RWY Througput with the OPT TSATS are:")
            print(simulation_OPT.runway_throughput[-1])
            #Comparing TSATS before and after Optim:
            print("The FCFS TSATS are:")
            print(simulation_FCFS.TSATs)

            print("The OPT TSATS are:")
            print(simulation_OPT.TSATs)

            print("The FCFS ATOTs are:")
            print(simulation_FCFS.ATOTs)

            print("The OPT ATOTs are:")
            print(simulation_OPT.ATOTs)

            print("The sorted ATOTs are:")
            print(np.sort(simulation_OPT.ATOTs))


            print("The FCFS wait times for deice are")
            print(simulation_FCFS.wait_times_for_deice_queue)

            print("The OPT wait times for deice are")
            print(simulation_OPT.wait_times_for_deice_queue)
            
            print("The FCFS departure delay times from gate")
            print(simulation_FCFS.dep_delay_times)

            print("The OPT departure delay times from gate")
            print(simulation_OPT.dep_delay_times)

           
            print("The FCFS departure total delay cost from gate")
            print(simulation_FCFS.total_delay_cost)

            print("The OPT departure total delay cost from gate")
            print(simulation_OPT.total_delay_cost)
        
            print("The FCFS total wait cost")
            print(simulation_FCFS.total_wait_cost_queue)

            print("The OPT total wait cost")
            print(simulation_OPT.total_wait_cost_queue)

            
            print("The FCFS total cost")
            print(simulation_FCFS.total_cost)

            print("The OPT total cost")
            print(simulation_OPT.total_cost)
       
        
        if montecarlo:
            #Run a new Monte Carlo simulation with the optimized TSATs with CRN Test seed:
            np.random.seed(crn_seed_test)
            random.seed(crn_seed_test)
            if crn_use_on_test:
                crn_seed_test_2 = crn_seed_test
            else: #Different seed for those two runs (does not make sense)
                crn_seed_test_2 = crn_seed_test+10 #just to make a different seed number.

            mc_simulation_opt = MonteCarloSimulation(M, flight_schedule, deicing_servers, throughput_cost, policies, printout,sim_mode,max_sim_time, crn_seed_test_2,antithetic,best_TSAT)
            mc_simulation_opt.run()

            if plots:

                # Create the figure and axis
                fig, ax = plt.subplots()

                if antithetic:
                    plotname = name +f"_mc_as"
                else:
                    plotname = name +f"_mc_cr"

                mc_simulation_FCFS.plot_running_average(burn_in,objective,plotname, ax = ax, label="FCFS")
                mc_simulation_opt.plot_running_average(burn_in,objective,plotname, ax = ax, label="OPT")

                fig.savefig(plotname)
                plt.show()

                plt.clf() # Clears the plot

                # Create the figure and axis
                fig, ax = plt.subplots()

                # Compare two policies:
                plotname_comp = plotname +f"_imp"

                mc_simulation_compare = MonteCarloSimulation(M, flight_schedule, deicing_servers, throughput_cost, policies, printout,sim_mode,max_sim_time, crn_seed_test,antithetic)
                #mc_simulation_compare.run()
                mc_simulation_compare.compare_policies(None, best_TSAT, seed_1= crn_seed_test, seed_2=crn_seed_test_2,objective=objective)

                print("Comparing two policies")
         
                mc_simulation_compare.plot_objective_diff(burn_in,plotname_comp,ax)

                #Save compare plot:
                plotname_comp = plotname +f"_compare"
                #plotname_comp = plotname +f"_compare_FCFS_vs_OPT"
                fig.savefig(plotname_comp)
                plt.show()

                #QQ plots and Histogram of diff:
                mc_simulation_compare.objective_diff_plots()

                #QQ and Hist for minibatches:
                M_batch = 32
                N= 1000 #Sample means collected
                mc_simulation_compare.minibatch_diff_plots(M_batch,N)


            if printout:
                if objective =="throughput":
                    print("Comparing Monte Carlo run with FCFS and OPT")
                    print(f"Example: {name} MC Simulation {M} runs FCFS Mean RWY Throughput")
                    print(mc_simulation_FCFS.mean_rwy_throughput)
                    print(f"Example: {name} MC Simulation {M} runs OPT Mean RWY Throughput")
                    print(mc_simulation_opt.mean_rwy_throughput)

                elif objective =="makespan":
                    print("Comparing Monte Carlo run with FCFS and OPT")
                    print(f"Example: {name} MC Simulation {M} runs FCFS Mean Makespan")
                    print(mc_simulation_FCFS.mean_makespan)
                    print(f"Example: {name} MC Simulation {M} runs OPT Mean Makespan")
                    print(mc_simulation_opt.mean_makespan)
                
                else:
                    print(f"Example: {name} MC Simulation {M} runs FCFS Mean Total Cost")
                    print(mc_simulation_FCFS.mean_total_cost)
                    print(f"Example: {name} MC Simulation {M} runs OPT Mean Total Cost")
                    print(mc_simulation_opt.mean_total_cost)

                    
    """PLOTS:"""
    if plots:
        plot_name_FCFS = f"{name}_FCFS"
        simulation_FCFS.plot_flight_times(plot_name_FCFS)
        simulation_FCFS.plot_takeoffs_over_time(plot_name_FCFS)
        if TSAT_optimization:
            plot_name_OPT = f"{name}_OPT"
            simulation_OPT.plot_flight_times(plot_name_OPT)
            simulation_OPT.plot_takeoffs_over_time(plot_name_OPT)

    """Saves logs of single FCFS and OPT runs to file"""
    if not montecarlo:
        filename_FCFS = f"log_of_sim_{name}_FCFS.csv"
        simulation_FCFS.save_log_to_csv(filename_FCFS)
    if TSAT_optimization:
        filename_OPT = f"log_of_sim_{name}_OPT.csv"
        simulation_OPT.save_log_to_csv(filename_OPT)

    # End timing
    end_time = time.time()
    execution_time = end_time - start_time
    # Print execution time in seconds
    print(f"DONE OPT. Exec time of the optimization with Optimizer {optimizer_method} of {name} is: {execution_time_opt:.4f} sec")
   
if __name__ == "__main__":
    """Runs this example:"""
    run_example(flight_schedule,deicing_servers,throughput_cost,crn_use_on_train,crn_use_on_test,crn_seed_train,crn_seed_test,optim_seed,
                sim_mode, opt_mode, policies, 
                montecarlo, M_runs_sim,M_runs_opt, 
                TSAT_optimization, optimizer_method, 
                printout, 
                plots, 
                name,antithetic,surrogat,buffer_optim,buff)
       


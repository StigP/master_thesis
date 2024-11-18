"""Version 110 optimizer algorithms"""

import os
import simpy
import time
import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

import heapq
import textwrap # For LaTex outputs
from fpdf import FPDF # For PDF output report.
from tabulate import tabulate 
from functools import wraps

#For Scipy Optimizing:
from scipy.optimize import dual_annealing #Generalized Simulated Annealing.
from scipy.optimize import differential_evolution
from scipy import stats

import itertools #for iterations.

#Own imports:
from simulation import FlightSimulation, MonteCarloSimulation
from sorting import FlightOptimizer #Own optimizer
from utils import milp_separation_solver, low_bound_deice_with_sep, deice_padlist, separation_time, sum_deice_pad_totals, sort_deice_pads_by_total, lognormal_survival_time



class TSATOptimization:
    def __init__(self, flight_schedule, deicing_servers, throughput_cost,policies,printout,max_sim_time,M,mode,optimizer, objective = "total_cost",max_optimizer_time=4, stop_at_bound = True,optim_seed = 123, crn_seed_train=123, crn_use_on_train = True, antithetic=True,surrogat=False,buffer_optim=None,buff=1):
        
        """
        Initializes the optimizer with the provided parameters.
        
        Parameters:
            flight_schedule (dict): The flight schedule as developed in utils.py.
            deicing_servers (int): The number of deicing servers available.
            throughput_cost (float): The cost of runway utilization in EUR.
            policies (list): The policy of for management of flights after deice.
            printout (bool): If True, prints detailed event logs and onscreen information.
            max_sim_time (float): The maximum simulation time allowed.
            M (int): The number of Monte Carlo runs.
            mode (str): Det or Sto: Either deterministic or stochastic optimization.
            optimizer (str): The optimizer type to be used.
            objective (str): The objective function to be optimized.
            max_optimizer_time (float): The maximum time allowed for the optimizer.
            stop_at_bound (bool): Whether to stop the optimizer if a bound is reached (default: True).
            optim_seed (int): The seed for random number generation used by the optimizer (default: 123).
            crn_seed_train (int): The seed for CRN (Common Random Numbers) during training (default: 123).
            crn_use_on_train (bool): Whether to use CRN during training (default: True).
            antithetic (bool): Whether to use antithetic variates for variance reduction (default: True).
            surrogat (bool): Whether to use a surrogate model for optimization (default: False).
            buffer_optim (any): Parameters related to buffer optimization (default: None).
            buff (int): The buffer value (default: 1).
        """

        # Seed management:
        self._optim_seed = optim_seed # Set to reproduce stochastic optimizer results
        self._mode = mode

        # Initialize separate random generator for the optimizer:
        self._rng_optim = np.random.default_rng(seed=self._optim_seed)  # For optimizer stochastics
       
        #Antithetic runs in optimizer. Default True
        self._antithetic = antithetic 

        #True of false if crn on train. Default True.
        self._crn_use_on_train = crn_use_on_train

        self._crn_seed_train = crn_seed_train #Same stream of CRN_train random numbers for each optimizer runs in Monte Carlo
        
        self.crn_seed_valid = self._crn_seed_train +10e9 # Seed for validation.
        
        self.M = M

        #Monte Carlo Batch runs:
       
        self.M_batch = min(self.M,32) # We set minimum M batch at 32
        # Number of Full batch size:
        self.M_batch_number_full = self.M//self.M_batch
        self.current_batch_partition = 1
        self.surrogat = surrogat


        self._buffer_policy = buffer_optim
        self._buff = buff

        if self._buffer_policy == "min":
            self._buffer_minutes = self._buff
            self._bufferoptimization = True
        elif self._buffer_policy == "prob":
            self._bufferprob = self._buff
            self._bufferoptimization = True
        
        else: #Fixed buffer in minutes - no optim
            self._buffer_minutes = buff
            self._bufferoptimization = False #Deactivates buffer optim

        # Init the flight schedule and Airport:
        self.flight_schedule = flight_schedule
        self.deicing_servers = deicing_servers

        self.number_flights = len(self.flight_schedule)
       
        # Init the simulation_train:
        self.simulation_train = MonteCarloSimulation(M, self.flight_schedule, self.deicing_servers, throughput_cost, policies, printout,mode,max_sim_time, self._crn_seed_train,self._antithetic, new_TSATs=None)
      
        
        self.throughput_cost = throughput_cost
        self.max_sim_time = max_sim_time
        self.printout = printout

        self.optimizer = optimizer #The selected optimizer algo.
        self.objective = objective #Throughput or total cost or makespan
        
        #For results report - Init at None:
        self._best_perm = None
        self._best_padlist = None
        self._padload = None

        
        self._FCFS_TSATs = self.simulation_train.TSATs
        self._FCFS_TSAT_delay = np.zeros_like(self._FCFS_TSATs)
        
        self.dimensions = len(self._FCFS_TSATs)
        self.iterations_best_makespan = []
        self.iterations_best_throughput = []
        self.iterations_best_waitcost = []
        self.iterations_best_delaycost = []
        self.iterations_best_totalcost = []
        self.iterations_lower_bound = []

        #Init a full best objective array at infinity:
        self._best_objective_arr = [np.inf] *self.M
        self._best_objective = np.inf
        
        #Store the naive current best solution (FCFS) arrays:
        
        # List of elite solutions (promising candidates) and rejected solutions:
        self._solution_cache = {} # Dictionary of all solutions tested.
        
        self._elite_solutions_set = set()
        self._elite_solutions = []
        self._elite_solution_values =[]
        self._rejected_solutions = []
        self._rejected_solution_values =[]
        self._num_elite_solutions = 0 #Init at zero
        self._num_rejected_solutions = 0 #Init at zero

        #Store runtimes:
        self._runtime = []
        self.start_time = time.time()  # Reset start times


        #Stop each optimizer in callback if max time has elapsed or bound has been reached:
        self._max_optimizer_time = max_optimizer_time
        self._stop_at_bound = stop_at_bound
        
        """For calculating throughput bounds:"""
        self.deice_workload = sum(flight['deice_expected_ADIT'] for flight in self.flight_schedule)
        self.earliest_at_deice = min(flight['expected_ERZT'] for flight in self.flight_schedule)
        
        # Find the lowest 'expected_ERZT' values that can go to deice first:
        self.lowest_ERZT_for_deice = heapq.nsmallest(self.deicing_servers, (flight['expected_ERZT'] for flight in self.flight_schedule))
        
        self.shortest_time_taxi_from_deice =  min(flight['exp_taxi_from_deice'] for flight in self.flight_schedule)
        
        self.earliest_at_rwy = min(flight['expected_ETOT'] for flight in self.flight_schedule)
        self.latest_at_rwy = max(flight['expected_ETOT'] for flight in self.flight_schedule)
        
        self.upper_bound_rwy_throughput() #calculates bounds at init.
        
        if self.objective=="total_cost":
            self.lower_bound_total_cost()


        # Define the bound:
        if self.objective =="throughput":
            self._bound_value = - self._throughput_bound
        
        if self.objective =="total_cost":
            self._bound_value = self.lowbound_cost

        if self.objective =="makespan":
            self._bound_value = self._lower_bound_makespan
        
        
    
    def adjust_tsats_for_permutation(self, flights):
        """
        Input is a set of fligths assigned to a particular deice server.

        Assigns TSATs to each flight assigned to a deice server
        by matching the finish times of previous flight and buffer size
        -TOBT is target off block time, the earliest the flight can leave gate 
        - EXOT. we let here be the taxi time from gate to deice or runway.
        - expected_ERZT is expected time at deice
        - ECZT: Expected deice commence time
        - buffer_EXOT is the buffer time allocated to arrive early at deice.
        
        Returns:
        - A list of TSATs for this deice server
        
        """
        
        tsat_list = []
        prev_flight = None # Will disregard flights that do not deice.
        
        for i, flight in enumerate(flights):
            flight['TSAT'] = flight['TOBT'] # Resets TSAT back to the original TOBT.

            if i == 0: # First flight
                flight['expected_ERZT'] = flight['TSAT'] + flight['expected_EXOT']
                flight['ECZT'] = flight['expected_ERZT']

                if flight['deice_expected_ADIT']>0:
                    prev_flight = flight
                else:
                    self._prev_nodeice_flight = flight
                    self._nodeice_last_TSAT = flight['TSAT']
            else:
                
                if flight['deice_expected_ADIT']>0:
                    
                    # Buffer times:
                    if self._buffer_policy == "prob":
                        
                        buffer_prob = self._bufferprob
                        if prev_flight is not None:
                            prev_adit_survival_t = lognormal_survival_time(prev_flight['deice_expected_ADIT_mu'],prev_flight['deice_expected_ADIT_sigma'],buffer_prob)
                            flight['buffer_EXOT'] = max(0,prev_flight['deice_expected_ADIT'] - prev_adit_survival_t)
                        
                    else: #Integer fixed minutes buffer time
                        
                        flight['buffer_EXOT'] = self._buffer_minutes

                    if prev_flight is None:
                        flight['expected_ERZT'] = flight['TSAT'] + flight['expected_EXOT']
                        flight['ECZT'] = flight['expected_ERZT']
                    else:
                        flight['expected_ERZT'] = max(flight['TSAT'] + flight['expected_EXOT'], prev_flight['expected_AEZT']-flight['buffer_EXOT'])
                        flight['ECZT'] = max(flight['TSAT'] + flight['expected_EXOT'], prev_flight['expected_AEZT'])
                    
                    #TSATS rounded to nearest minute if a prob based buffer:
                    flight['TSAT'] = round(flight['expected_ERZT'] - flight['expected_EXOT'])
                    
                    prev_flight = flight
                    
                else: #for flights bypass deice:
                    sep = 1
                    
                    if self._nodeice_last_TSAT is None:
                        flight['ECZT'] = flight['TSAT'] + flight['expected_EXOT']

                    else:
                        sequence_flts = self._prev_nodeice_flight,flight
                        sep = int(separation_time(sequence_flts))
                        
                        flight['ECZT'] = max(self._nodeice_last_TSAT + sep + flight['expected_EXOT'], flight['TSAT'] + flight['expected_EXOT'])
                    
                    flight['TSAT'] = round(flight['ECZT'] - flight['expected_EXOT'])
                    
                    # Stores the last TSAT of a no deice flight:
                    self._prev_nodeice_flight = flight
                    self._nodeice_last_TSAT = flight['TSAT']
                    
            flight['expected_AEZT'] = flight['ECZT'] + flight['deice_expected_ADIT']
            
            tsat_list.append({'flight_id': flight['flight_id'], 'TSAT': flight['TSAT'], 'index': flight['original_index']})
             
        return tsat_list

  
    def run_optimizers(self, optimizers):
       
        """
        Executes multiple optimizers and records the results, including runtime and performance metrics.

        It iterates over a dictionary of optimizers, runs each and records statistics from each run, 
        and determines the best-performing optimizer based on objetive and runtime.

        Parameters:
            optimizers (dict): A dictionary where keys are optimizer names (str) and values are 
                            optimizer instances to be executed.

        Returns:
            tuple: A tuple containing:
                - results (dict): A dictionary with optimizer names as keys and their performance metrics 
                as values.
                - best_optimizer (tuple): A tuple containing the name of the best optimizer and its 
                result dictionary.
        """
        

        results = {}
        for name, optimizer in optimizers.items():
            print(f"Running optimizer: {name}")
            self.optimizer = optimizer

            #Restart each optimizer with same seed
            self._rng_optim = np.random.default_rng(seed=self._optim_seed)  # For optimizer stochastics
    
            #Store runtimes:
            self._runtime = []
            self.start_time = time.time()  # Reset start times

        
            #Init a full best objective array at high values:
            self._best_objective_arr = [1e20] *self.M
            self._best_objective = 1e20
             
            # List of elite solutions (promising candidates) and rejected solutions:
            self._solution_cache = {} # Dictionary of all solutions tested.
            self._fcfs_diff_cache = {} #Dict of all diff from FCFS tested.
            self._elite_solutions_set = set()
            self._elite_solutions = []
            self._elite_solution_values =[]
            self._rejected_solutions = []
            self._rejected_solution_values =[]
            self._num_elite_solutions = 0 #Init at zero
            self._num_rejected_solutions = 0 #Init at zero
            self._num_duplicate_solutions = 0 #Init at zero
            self._num_Cx_calls = 0 # Calls to a single simulation
            self._num_Cx_MC_calls = 0 # Calls to a complete MC run
            
            
            #Init with the FCFS solution:
            self._new_value_diff_FCFS = 0 #Diff array from FCFS for noisy SA
            self._FCFS_objective_array = None
            self._FCFS_evaluated = False

            # The FCFS objective value:
            self._FCFS_objective_val = self.evaluate(self._FCFS_TSAT_delay)
           

            #Associated FCFS values if not the objective - replaced below:
            self._FCFS_best_throughput = self.simulation_train.mean_rwy_throughput
            self._FCFS_best_makespan = self.simulation_train.mean_makespan
            self._FCFS_best_totalcost = self.simulation_train.mean_total_cost

            if self.objective== "makespan":
                self._FCFS_best_makespan = self._FCFS_objective_val
            
            if self.objective== "total_cost":
                self._FCFS_best_totalcost = self._FCFS_objective_val

            if self.objective== "throughput":
                self._FCFS_best_throughput = self._FCFS_objective_val

            
            # Init Buffer optimizers for stochastic buffer:
            
            if self._buffer_policy == "min":
                #self.bufferoptim_by_minutes = True
                self._buffer_minutes = self._buff
                #self.buffer_optimization_fixed_minutes = buff
                self._bufferoptimization = True
            elif self._buffer_policy == "prob":
                self._bufferprob = self._buff
                self._bufferoptimization = True
        
            else: #Fixed buffer in minutes - no optim
                self._buffer_minutes = self._buff
                self._bufferoptimization = False #Deactivates buffer optim
        
            self._modify_buffer = False
            self._last_buffer_move_improved = True #Flag for direction
            
            # Buffer poliy prob:
            self.bufferprob_lower_bound = 0.5
            self.bufferprob_upper_bound = 0.9999
            self.buffer_bound_range = self.bufferprob_upper_bound - self.bufferprob_lower_bound
            self.buffer_increment = 0.01 #Initial step size and direction.

            # Buffer policy in fixed minutes:
            if self._buffer_policy == "min":
                self._buffermin_lower_bound = 0
                self._buffermin_upper_bound = 20
                self.buffer_bound_range = self._buffermin_upper_bound - self._buffermin_lower_bound

            self.last_move_direction = "up" #Init
            self._buffer_list = [] #Tracks buffer changes

            # Lower bound for a buffer based on prob:
            if self.objective == "makespan" or "throughput":
                # For prob buffer policy
                self.bufferprob_lower_bound = 0.6
                

            # Iterations lists:
            self.iterations_best_makespan = []
            self.iterations_best_throughput = []
            self.iterations_best_waitcost = []
            self.iterations_best_delaycost = []
            self.iterations_best_totalcost = []
            self.iterations_lower_bound = []

            #Store runtimes:
            self._runtime = []
            self.start_time = time.time()  # Reset start times


            # Run the optimization
            best_TSAT, optimized_objective = self.optimize_TSATS()
            
            
            # Sorted indices by np.lexsort to sort by best_TSAT values
            # and then by the index order:
            sorted_indices = np.lexsort((np.arange(len(best_TSAT)), best_TSAT))

            # One-based indexing
            offblock_sequence_flts = sorted_indices + 1
            
            # Sum of ADIT deice load assigned to each pad:
            if self._best_perm is not None:
                self._padload = [sum(flight['deice_expected_ADIT'] for flight in pad) for pad in self._best_perm]  
                
                #Sorts on totals:
                sorted_deice_pads, sorted_deice_pad_totals = sort_deice_pads_by_total(self._best_perm, self._padload)
                self._best_padlist = deice_padlist(sorted_deice_pads)
                
                #Rename flights for shorter report (keeps two last digits only):
                abbrev_padlist = [[f'{flight[-2:]}' for flight in pad] for pad in self._best_padlist]
                
                self._best_padlist = abbrev_padlist
                self._padload = sorted_deice_pad_totals
                

            # Time to best (s):
            # Find the runtime at which each optimizer first reached the best result
            if self.objective == "throughput":
                # Look for the first occurrence of the maximum throughput
                best_time_index = next(i for i, val in enumerate(self.iterations_best_throughput) if val == optimized_objective)
            
            elif self.objective == "makespan":
                # Look for the first occurrence of the min makespan
                best_time_index = next(i for i, val in enumerate(self.iterations_best_makespan) if val == optimized_objective)

            else:
                # Look for the first occurrence of the minimum total cost
                best_time_index = next(i for i, val in enumerate(self.iterations_best_totalcost) if val == optimized_objective)

            best_runtime = self._runtime[best_time_index]

            # All the values with the optimized TSAT:
            # Will only use the one that really is the objective.
            makespan = optimized_objective
            totalcost = optimized_objective
            throughput = optimized_objective

            # Metrics:
            # Gap to Bound:
            bound_gap = abs(optimized_objective - self._bound_value)    
            epsilon = 1e-5 #Avoids div by zero if bound is 0
            relative_pct_bound_gap = 100*bound_gap/(self._bound_value + epsilon)
            
        
            # Gap to FCFS:
            fcfs_gap = abs(optimized_objective - self._FCFS_objective_val)
            relative_pct_fcfs_gap = 100*fcfs_gap/self._FCFS_objective_val
           

            # Store the results for each optimizer and metrics:
            results[name] = {
                "best_TSAT": best_TSAT,
                "FCFS_TSAT": self._FCFS_TSATs,
                "offblock_seq": offblock_sequence_flts,
                "FCFS_objective": self._FCFS_objective_val,
                "optimized_objective": optimized_objective,
                "Bound": self._bound_value,
                "FCFS_gap": fcfs_gap,
                "%\imp":relative_pct_fcfs_gap,
                "Bound_gap": bound_gap,
                "%\gap":relative_pct_bound_gap,
                "FCFS_makespan": self._FCFS_best_makespan,
                "FCFS_totalcost": self._FCFS_best_totalcost,
                "OPT_makespan": makespan.copy(),
                "OPT_totalcost": totalcost.copy(),
                "iterations_best_throughput": self.iterations_best_throughput.copy(),
                "iterations_best_totalcost": self.iterations_best_totalcost.copy(),
                "runtime": self._runtime.copy(),
                "iterations_best_makespan": self.iterations_best_makespan.copy(),
                "tot_ADIT": self.deice_workload,
                "best_permutation": self._best_perm,
                "best_padlist":self._best_padlist,
                "ADIT_per_pad": self._padload,
                "best_runtime": best_runtime  # Store the time when the best value was first achieved
            }

            # Resets for the next optimizer.
            self.iterations_best_throughput.clear()
            self.iterations_best_totalcost.clear()
            self.iterations_best_makespan.clear()
            self._runtime.clear()
            self._best_perm = None
            self._best_padlist = None
            self._padload = None


        # Find the best optimizer based on result, and for ties, on runtime to best:
        if self.objective == "throughput":
            best_value = max(result["optimized_objective"] for result in results.values())
        else:
            best_value = min(result["optimized_objective"] for result in results.values())
 

        # Get all optimizers that reached the best value
        best_optimizers = [name for name, result in results.items() if result["optimized_objective"] == best_value]

        # If there are multiple optimizers with the same best value, the best is the one that reached it first
        if len(best_optimizers) > 1:
            sorted_best_optimizers = sorted(
                ((name, results[name]) for name in best_optimizers),
                key=lambda x: x[1]["best_runtime"]
            )
            best_optimizer = sorted_best_optimizers[0]
            second_best_optimizer = sorted_best_optimizers[1]

            # Print runtime comparison
            if best_optimizer[1]["best_runtime"] > 0:
                speedup_factor = second_best_optimizer[1]["best_runtime"] / best_optimizer[1]["best_runtime"]
                print(f"The best optimizer is {best_optimizer[0]} with a runtime of {best_optimizer[1]['best_runtime']:.4f} seconds, "
                    f"which is {speedup_factor:.1f}x faster than the second best ({second_best_optimizer[0]}).")
            else:
                print(f"The best optimizer is {best_optimizer[0]} with a runtime of zero")

        else:
            best_optimizer = (best_optimizers[0], results[best_optimizers[0]])
            print(f"The best optimizer is {best_optimizer[0]} with an optimized {self.objective} of {best_optimizer[1]['optimized_objective']:.4f} with a runtime of {best_optimizer[1]['best_runtime']:.4f} seconds")

        
        self.results = results
        self.best_optimizer = best_optimizer
        self._best_value = best_value

        return results, best_optimizer

 

    def evaluate(self, TSAT_delay):
        """Evaluates using adaptive Monte Carlo function calls to the DES sim of C(X) with CRN_train_seed"""
       
        tsat_delay_tuple = tuple(TSAT_delay)

        if tsat_delay_tuple in self._solution_cache:
            #Skips Monte Carlo - look up value from duplicate run
            self._num_duplicate_solutions +=1

            #Updates:
            self.iterations_best_throughput.append(self._best_objective)
            self.iterations_best_makespan.append(self._best_objective)
            self.iterations_best_totalcost.append(self._best_objective)
                
            # Track the runtime
            self.current_time = time.time()
            self._runtime.append(self.current_time - self.start_time)

            # Update FCFS diff for Sim Anneal new value:
            self._new_value_diff_FCFS = self._fcfs_diff_cache[tsat_delay_tuple]

            return self._solution_cache[tsat_delay_tuple]
        
        
        # TSAT for full simulation C(X):
        self.simulation_train.TSATs = self._FCFS_TSATs + TSAT_delay
        
        
        # Batches
        M_batch = self.M_batch
        M_max = self.M  # Set maximum number of runs
        
        M_partitions_full = self.M_batch_number_full # Number of full size batch partitions.
        full_batch_size = True # Changes to False for last run if there is a partial batch
        #The part partition using CRN this evaluation will start on:
        sel__start_partition = self.current_batch_partition -1
        
        # Increments start for the next evaluation:
        self.current_batch_partition = (self.current_batch_partition%self.M_batch_number_full)+1
       
        total_samples = 0  # Keep track of the total number of evaluated samples
        remaining_samples = M_max # Keep track of remaining

        can_array_parts = [] # To build the entire candidate array appending from the batches
        can_array_full = np.zeros(M_max) #Builds the candidate with correct indexing if it becomes new best
        
        diff_array = [] # To accumulate differences, used for the batchwise evaluation of reject or continue
        diff_can_vs_FCFS = [] # To evaluate batchwise diff from FCFS array (for Sim Ann)

        best_arr = self._best_objective_arr

        
        while remaining_samples > 0:
            
            if remaining_samples<M_batch:
                M_batch = remaining_samples
                full_batch_size = False
                
            #Partitions:
            sel__start_partition = (sel__start_partition%self.M_batch_number_full)+1
            
            #Index for start and end:
            if full_batch_size:    
                start_index = (sel__start_partition - 1) * M_batch
                end_index = start_index + M_batch

            else: #partial last batch which is always done at the end:
                start_index = (total_samples)-1
                end_index = M_max-1
                
            #Counter for samples and partitions:
            total_samples += M_batch
            remaining_samples = M_max - total_samples

            #Update seeds for each batch based on partition:
            crn_batch_seed = self._crn_seed_train + sel__start_partition #Ensures same CRN for batch evaluations
            
            # Run the full Monte Carlo simulation:
            self.simulation_train.run(M_runs = M_batch, seedchange=crn_batch_seed)
            self._num_Cx_calls += M_batch
            self._num_Cx_MC_calls += 1
            
            # Choose the correct array based on objective:
            if self.objective == "makespan":
                candidate_batch = self.simulation_train.makespan_arr
                
            elif self.objective == "total_cost":
                candidate_batch = self.simulation_train.totalcost_arr
                
            elif self.objective == "throughput":
                candidate_batch = - self.simulation_train.throughput_arr
            
            # Insert batches into full array based on partition start and end index:
            can_array_full[start_index:end_index] = np.copy(candidate_batch)

            # Add batch at the can array parts:
            can_array_parts.extend(candidate_batch) #Accumulate a list of the candidate batches for mean stats.

            #Objective value:
            objective_value = np.mean(can_array_parts)
            
            diff_batch_arr = can_array_full[start_index:end_index] - best_arr[start_index:end_index]
            if self._FCFS_objective_array is not None:
                diff_can_vs_FCFS_batch = can_array_full[start_index:end_index] - self._FCFS_objective_array[start_index:end_index]
                diff_can_vs_FCFS.extend(diff_can_vs_FCFS_batch)
            
                #For Simulated Annealing with noise:
                new_value_diff_FCFS = np.mean(diff_can_vs_FCFS)
             
                self._new_value_diff_FCFS = new_value_diff_FCFS
                
            #Extend the diff_array with this batch:
            diff_array.extend(diff_batch_arr)

    
            # T-test now if diff_array has more than 20 samples:
            if self.M_batch>20:
                mean_diff = np.mean(diff_array)
                std_diff = np.std(diff_array, ddof=1)
                if std_diff==0:
                    #print("Possible duplicate run")
                    T_statistic = 0
                    
                else:
                    T_statistic = mean_diff / (std_diff / np.sqrt(total_samples))
            
                # Calculate the critical t-value for 95% confidence
                confidence_level = 0.95
                t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df=total_samples - 1)

                # If candidate is assumed worse, stop further evaluations:
                if (T_statistic > t_critical):
                 
                    self._rejected_solutions.append(TSAT_delay)
                    self._rejected_solution_values.append(objective_value)
                    self._num_rejected_solutions += 1

                    #Update all:
                    self.iterations_best_throughput.append(self._best_objective)
                    self.iterations_best_makespan.append(self._best_objective)
                    self.iterations_best_totalcost.append(self._best_objective)
                        
                    #Track the runtime
                    self.current_time = time.time()
                    self._runtime.append(self.current_time - self.start_time)

                    # Cache the solution:
                    self._solution_cache[tsat_delay_tuple] = (objective_value)
                    self._fcfs_diff_cache[tsat_delay_tuple] =(new_value_diff_FCFS)

                    return objective_value
            
    
        # Check if the candidate solution is equal or better than current best:
        candidate_is_not_worse = False
        if objective_value <= self._best_objective:
            candidate_is_not_worse = True

    
        # If candidate is better or equal to current best, update the best metrics
        if candidate_is_not_worse:

            # Update current best obj val and the entire array from MC:
            self._best_objective = objective_value
            self._best_objective_arr = np.copy(can_array_full)
            self.current_best_TSAT = self.simulation_train.TSATs

            tsat_delay_tuple = tuple(TSAT_delay)
          
            if tsat_delay_tuple not in self._elite_solutions_set:
                self._elite_solutions.append(TSAT_delay)
                self._elite_solutions_set.add(tsat_delay_tuple)
                self._elite_solution_values.append(objective_value)
                self._num_elite_solutions += 1

            if self.printout:
                print(f"Candidate's solution {TSAT_delay} with {objective_value} is evaluated to be better.")
                print(f"Number of elite candidates: {self._num_elite_solutions}")
                print(f"Number of rejected candidates: {self._num_rejected_solutions}")
                

        else: #Candidate is worse:
            self._rejected_solutions.append(TSAT_delay)
            self._rejected_solution_values.append(objective_value)
            self._num_rejected_solutions += 1
            
        
        #Update all:
        self.iterations_best_throughput.append(self._best_objective)
        self.iterations_best_makespan.append(self._best_objective)
        self.iterations_best_totalcost.append(self._best_objective)
            
        # Track the runtime
        self.current_time = time.time()
        self._runtime.append(self.current_time - self.start_time)
        
        
        self._solution_cache[tsat_delay_tuple] = (objective_value)
        
        if not self._FCFS_evaluated:
            self._FCFS_objective_array = can_array_full
            new_value_diff_FCFS = 0

            self._FCFS_evaluated = True

        self._fcfs_diff_cache[tsat_delay_tuple] =(new_value_diff_FCFS) 

        return objective_value


    def tailored_search(self, objective_function, local_search=False, callback=None):
        """The main algorithm for Tailored Search (TS) 
        - It reuses the SA algo where it turns off local SA, effectively disabling SA"""
        SA_start_iter = np.inf #disables SA
        best_solution, best_value = self.simulated_annealing(objective_function, local_search, SA_start_iter, callback)
        
        return best_solution, best_value
    

    def simulated_annealing(self, objective_function, local_search=False, SA_start_iter=0, callback=None, 
                            initial_temperature=1e5, cooling_rate=0.999, iter_per_temp=10, 
                            max_global_iter=10000000, max_local_iter=10000000, plotting=False,auto_temp=False,random_walk=False,greedy_descent=False,target_accept_prob_init=0.90,auto_cool=False):
        
        """
        Simulated Annealing (SA) optimization algorithm for flight deicing scheduling.
        
        Args:
        objective_function (function): The function to minimize, which calculates the objective value for a given solution.
        local_search (bool, optional): Whether to use local search in the input. Default is False.
        SA_start_iter (int, optional): The global iteration at which to start the simulated annealing process. Default is 0.
        callback (function, optional): A callback function to evaluate and potentially stop early. Default is None.
        initial_temperature (float, optional): The initial temperature for SA cooling schedule. Default is 1e5.
        cooling_rate (float, optional): The rate at which to cool the temperature. Default is 0.999.
        iter_per_temp (int, optional): The number of iterations to perform before cooling the temperature. Default is 10.
        max_global_iter (int, optional): The maximum number of global iterations (restarts). Default is 100000.
        max_local_iter (int, optional): The maximum number of local iterations per SA process. Default is 10000.
        plotting (bool, optional): Whether to plot the progress of the algorithm. Default is False.
        auto_temp (bool, optional): Whether to readjust initial temperature based on initial 10 runs. Default is False

        Returns:
        tuple: 
            best_solution (list): The best TSAT solution found by the algorithm.
            best_value (float): The objective value of the best solution.
        """
        
        target_accept_prob_final = 0.01 # To adjust cooling schedule.

        # Initialize optimizer, variables, and random permutation generator
        optimizer = FlightOptimizer(self.flight_schedule, self.deicing_servers, local_search, self._optim_seed)
        perm_copy = optimizer.generate_random_permutations()  # yields TS permutations
        
        global_iterations = 0
        local_iterations = 0
        total_iterations = 0

        acceptance_probability = 0

        # Target acceptance prob for init runs temperature adj:
        if auto_temp:
            self._target_acc_prob_init = target_accept_prob_init
            #delta Energy list (for temp adjust after init runs):
            delta_E = []
            
        # Temperature:
        self._initial_temperature = initial_temperature

        if auto_cool:
            # Adjusts coolrate based on the runtime allocated and resets for local iterations

            self.cool_rate_is_adjusted = False
            self._target_Temp_final = 1e-12

            #New cooling rate based on reaching T_final in each local iterations:
            rem_temp_updates = np.ceil(max_local_iter/iter_per_temp)
            cooling_rate = np.exp(np.log(self._target_Temp_final/self._initial_temperature)*1/rem_temp_updates)
            
        # Init values:
        best_value = np.inf #Objective value
        best_diff_value = np.inf # Difference from FCFS
        best_perm = None
        best_solution = None

        # Iteration lists for plotting
        global_iteration_list = [] #Outer loop, counts the restarts of init sol
        local_iterations_list = [] #Inner SA loop - counts SA iterations from one init sol
        total_iterations_list = [] # Sum of all iterations

        best_values_list = []
        temperature_list = []
        accept_prob_list = []
        accepted_values_list = []
        

        temperature = self._initial_temperature
        iter_per_temp_0 = iter_per_temp
        beta_rate =1.2 #Fixed. This increases the iterations at lower temp

        # Loop over generated permutations
        for perm in perm_copy:
            acceptance_probability = 0 #Reset for outer non SA loop
            self._nodeice_last_TSAT = None
            combined_tsats = []

            # Calculate TSAT for the current permutation
            for pad in perm:
               
                tsat_list = self.adjust_tsats_for_permutation(pad)
                combined_tsats.extend(tsat_list)
            tsat_array = self.get_sorted_tsat_values([combined_tsats])
        
            global_iterations += 1
            total_iterations +=1

            # Boolean to start SA:
            Start_SA = global_iterations > SA_start_iter

            # Calculate objective value for the current solution
            tsat_array = np.array(tsat_array).flatten()
            candidate_solution = tsat_array - self._FCFS_TSATs
            
            candidate_value = objective_function(candidate_solution)

            candidate_diff_value = self._new_value_diff_FCFS

            candidate_perm = [pad[:] for pad in perm]  # deep copy

            # Update best solution if found
           
            if candidate_diff_value < best_diff_value:
                best_value = candidate_value
                best_diff_value = candidate_diff_value

                best_solution = candidate_solution
                acceptance_probability = 1 #For the non SA part.
                best_perm = [pad[:] for pad in candidate_perm]  # deep copy
                if self.printout:
                    print(f"New best solution at global iteration {global_iterations}")
                    print(f"Best value: {best_value}")

            # Store for plotting
            global_iteration_list.append(global_iterations)
            total_iterations_list.append(total_iterations)
            best_values_list.append(best_value)
            accepted_values_list.append(best_value)
            temperature_list.append(temperature)
            accept_prob_list.append(acceptance_probability)
            
            #Starts the Simulated Annealing:
            if Start_SA:
                # Reset temperature for each local iteration run
                temperature = self._initial_temperature
                #print("line 1516 new SA run local: temp:")
                #print (self._initial_temperature)
                
                iter_per_temp_0 = iter_per_temp

                for local_iteration in range(max_local_iter):
                    total_iterations +=1
                    local_iterations +=1
                   
                    # Generate a neighboring solution:
                    new_perm, new_TSAT_array = self.get_neighboring_solution_perm(candidate_perm)
                    new_solution = np.array(new_TSAT_array).flatten() - self._FCFS_TSATs
                    
                    new_value = objective_function(new_solution)
                    new_diff_value = self._new_value_diff_FCFS

                    
                    # To adjust init temperature based on delta_E and diff values:
                    if auto_temp:
                        #if len(delta_E)<20:
                        self._deltaE = new_diff_value - candidate_diff_value
                        delta_E.append(self._deltaE)
                                                    
                        if len(delta_E)==20: #After 20 init runs we adjust temp
                            
                            #max_dE = abs(np.max(delta_E))
                            
                            max_dE = np.percentile(np.abs(delta_E), 90) # To avoid outliers.
                            
                            target_temp = - max_dE/np.log(self._target_acc_prob_init)
                            
                            # Update target temp final:
                            self._target_Temp_final = - max_dE/np.log(target_accept_prob_final)
                            self._initial_temperature = target_temp
                            temperature = target_temp

                            # Recalculate the cooling rate if auto_cool is enabled
                            if auto_cool:
                                rem_temp_updates = np.ceil(max_local_iter / iter_per_temp)
                                cooling_rate = np.exp(np.log(self._target_Temp_final / self._initial_temperature) / rem_temp_updates)
                                print(f"Recalculated cooling rate: {cooling_rate}")
                                
                           
                    # Selects either extremes (RW or GD) or SA:
                    if random_walk:
                        # Locks acceptance at 1 for a RW
                        acceptance_probability = 1

                    elif greedy_descent:
                        # Locks acceptance at 0 for a GD
                        acceptance_probability = 0

                    else: #Sim Annealing:
                     
                    # Calculate acceptance probability and accept/reject new solution
                        acceptance_probability = self.calculate_acceptance_probability(candidate_diff_value, new_diff_value, temperature)
                        

                    if new_diff_value < candidate_diff_value or (self._rng_optim.random() < acceptance_probability):
                       
                        if self._bufferoptimization:
                            if new_diff_value<candidate_diff_value:
                                
                                self._last_buffer_move_improved = True
                                
                            else:
                                
                                self._last_buffer_move_improved = False
                              
                        candidate_solution = new_solution
                        candidate_value = new_value
                        candidate_diff_value = new_diff_value
                        
                        candidate_perm = [pad[:] for pad in new_perm]  # deep copy needed

       
                    # Update best solution if found
                    if candidate_value < best_value:
                        best_solution = candidate_solution
                        best_value = candidate_value
                        best_perm = [pad[:] for pad in candidate_perm]  # deep copy needed
                        self._best_perm = best_perm
                       

                        if self.printout:

                            print(f"New best solution by SA at global iterations {global_iterations}")
                            print(f"And at local iteration {local_iteration}")
                            print(f"Best TSAT delta solution: {best_solution}")
                            print(f"Best flights to deice pads solution: {deice_padlist(best_perm)}")
                            print(f"Best value: {best_value}")
                            print(f"Temp was: {temperature}")
                        
                            
                    # Store for plots:
                    global_iteration_list.append(global_iterations)
                    local_iterations_list.append(local_iterations)
                    total_iterations_list.append(total_iterations)
                    best_values_list.append(best_value)
                    accepted_values_list.append(candidate_value)
                    temperature_list.append(temperature)

                    accept_prob_list.append(acceptance_probability)
                    
                    # Cool down geometrically at intervals and update iter length
                    if auto_cool and not self.cool_rate_is_adjusted:
                        fraction_done = 0.05
                        if self._runtime[-1]>fraction_done*self._max_optimizer_time:
                            
                            rem_iter = total_iterations*(1-fraction_done)/fraction_done
                            rem_local_iter = max_local_iter - local_iteration
                            
                            if rem_iter<rem_local_iter:
                                rem_temp_updates = rem_iter/iter_per_temp

                                # New cooling rate:
                                cooling_rate = np.exp(np.log(self._target_Temp_final/temperature)*1/rem_temp_updates)
                                
                            self.cool_rate_is_adjusted = True
                            

                    if local_iteration > 0 and local_iteration % iter_per_temp_0 == 0:
                        temperature *= cooling_rate
                       
                        iter_per_temp_0 = math.ceil(iter_per_temp_0 * beta_rate)
                        
                    # Check in SA if callback requests to stop early:
                    if callback is not None:
                        stop = callback(best_solution, best_value)
                        if stop is not None and stop:
                            print(f"Callback function requested to stop early at global iteration {global_iterations} "
                                f"and local iteration {local_iterations} and total iteration {total_iterations}")
                            print(f"Total Monte Carlo calls: {self._num_Cx_MC_calls} which made {self._num_Cx_calls} simulation calls")
                            print(f" Duplicate runs: {self._num_duplicate_solutions}")

                            if plotting:
                                self.plot_results(total_iterations_list, best_values_list, temperature_list, accepted_values_list)
                            
                            #Store best permutation:
                            self._best_perm = best_perm
                            return best_solution, best_value

            # Check globally if callback requests to stop early:
            if callback is not None:
                stop = callback(best_solution, best_value)
                if stop is not None and stop:
                    print(f"Callback function requested to stop early at global iteration {global_iterations} "
                    f"and local iteration {local_iterations} and total iteration {total_iterations}")
                    print(f"Total Monte Carlo calls: {self._num_Cx_MC_calls} which made {self._num_Cx_calls} simulation calls")
                    print(f" Duplicate runs: {self._num_duplicate_solutions}")
                    
                    if plotting:
                        self.plot_results(total_iterations_list, best_values_list, temperature_list, accepted_values_list)
                    #Store best permutation:
                    self._best_perm = best_perm
                    return best_solution, best_value,

        
        if plotting:
            self.plot_results(total_iterations_list, best_values_list, temperature_list, accepted_values_list)
        #Store best permutation:
        self._best_perm = best_perm
        return best_solution, best_value



    def get_neighboring_solution_perm(self, candidate_perm):
        """
        Neighborhood Function for SA. The selected move is probabilistically
        assigned either as a pure move or a swap between two flights. Or a buffer optim move
        The deice pads selected for moves are either by random or for balancing the loads
        (balance move)
        """
        # Decide if buffermove:
        # Determine whether to modify the buffer based on the result of the last move
        if self._bufferoptimization: #Only if active
            if self._last_buffer_move_improved:
                self._modify_buffer = True
                
            else:
                # Random chance to modify the buffer if the last move didn't improve
                self._modify_buffer = self._rng_optim.random() < 0.1

                # Deactivate buffer modification if the range is small enough
                if self._buffer_policy == "prob" and self.buffer_bound_range < 0.02: 
                    self._bufferoptimization = False
                
                # Deactivate buffer modification if the range is small enough
                if self._buffer_policy == "min" and self.buffer_bound_range < 1:
                    
                    self._bufferoptimization = False

            # If buffer move:
            if self._modify_buffer:

                if self._buffer_policy == "prob":
                    
                    self.buffer_optimization_by_prob()

                if self._buffer_policy == "min":
                    
                    self.buffer_optimization_fixed_minutes()
                
                # Recalculate the new TSAT array after the buffer move
                combined_tsats = []
                self._nodeice_last_TSAT = None
                for pad in candidate_perm:
                    tsat_list = self.adjust_tsats_for_permutation(pad)
                    combined_tsats.extend(tsat_list)
                new_solution = self.get_sorted_tsat_values([combined_tsats])

                # If a buffer move was done we return candidate perm with the new TSATS:
                return candidate_perm, new_solution
        
        """If not a buffer move:"""
        # Generate a neighboring solution by randomly swapping two flights or moving one flight in the current solution
        selected_move_operation = self._rng_optim.random()
        balanced_move_operation = self._rng_optim.random()
        
        pad_totals = sum_deice_pad_totals(candidate_perm)
        
        index_max = np.argmax(pad_totals)
        index_min = np.argmin(pad_totals)

        pad_diff = np.max(pad_totals) - np.min(pad_totals)

        swap_threshold = 0.5 #Decides if swap or move
        a = 0.1 # Scaling parameter for the sigmoid function below.

        # Sigmoid. Decides if move or swap from index with highest load.
        balance_move_threshold =1- 1/(1+np.exp(-a*pad_diff)) 

        swap = selected_move_operation > swap_threshold  # Decides if swap or pure move.
        balance_move = balanced_move_operation > balance_move_threshold

        # Create a deep copy of candidate_perm
        new_perm = [pad[:] for pad in candidate_perm]
        
        #Decide on which index to move/swap:
        if balance_move:
            i = index_max
            j = index_min

        else: #Not a balance move:
            # Randomly choose two deicing pads (may be the same)
            i, j = self._rng_optim.choice(range(len(new_perm)), size=2, replace=True)
            
        # Ensure there is at least one flight in each selected pad after swap/move
        if swap:
            # Choose pads that have more than one flight to avoid emptying any pad
            if len(new_perm[i]) > 1 and len(new_perm[j]) > 1:
                k = self._rng_optim.integers(0, len(new_perm[i]))
                l = self._rng_optim.integers(0, len(new_perm[j]))
                
                # Swap flight k from deice pad i with flight l from deice pad j
                new_perm[i][k], new_perm[j][l] = new_perm[j][l], new_perm[i][k]

        else:
            # Move flight k from deice pad i to deice pad j
            # Ensure pad i has more than one flight before move:
            if len(new_perm[i]) > 1:
                k = self._rng_optim.integers(0, len(new_perm[i]))
                flight_to_move = new_perm[i].pop(k)
                new_perm[j].append(flight_to_move)
        
        # Recalculate the new TSAT array after the move or swap:
        combined_tsats = []
        self._nodeice_last_TSAT = None
        for pad in new_perm:
            tsat_list = self.adjust_tsats_for_permutation(pad)
            combined_tsats.extend(tsat_list)
            
        new_solution = self.get_sorted_tsat_values([combined_tsats])
        
        return new_perm, new_solution


    def calculate_acceptance_probability(self, candidate_value, new_value, temperature):
        """ SA acceptance prob"""
        if new_value < candidate_value:
            return 1.0
        return math.exp((candidate_value - new_value) / temperature)


    def brute_force_search(self, objective_function, dimensions, value_range, callback=None):
        """Brute Force search with lazy evaluation using a generator."""
        best_value = np.inf
        best_solution = None    
        value_ranges = [range(b[0], b[1] + 1) for b in value_range]

        # Generator for brute force that yields each possible solution lazily:
        def solution_generator():
            for values in itertools.product(*value_ranges):
                yield np.array(values)

        # Process each solution as it is generated
        for candidate_solution in solution_generator():
            candidate_value = objective_function(candidate_solution)
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Check if callback requests to stop early
            if callback is not None:
                stop = callback(best_solution, best_value)
                if stop is not None and stop:
                    print("Callback function requested to stop early")
                    return best_solution, best_value

        return best_solution, best_value


    def random_search(self, objective_function, dimensions, value_range, num_samples=1000000, callback=None):
        """Implements a RS Random Search in the full search space (V) - For benchmark"""
        best_value = np.inf
        best_solution = None
        
        # Generate random combinations of values
        for _ in range(num_samples):
            # Create a random combination within the value range for each dimension
            random_values = [self._rng_optim.integers(low=b[0], high=b[1] + 1) for b in value_range]
            
            candidate_solution = np.array(random_values)
            
            # Evaluate the candidate solution
            candidate_value = objective_function(candidate_solution)
            
            # Update the best solution if the current one is better
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution
            
            # Check if callback requests to stop early
            if callback is not None:
                stop = callback(best_solution, best_value)
                if stop is not None and stop:
                    print("Callback function requested to stop early")
                    return best_solution, best_value
                
        return best_solution, best_value


    def objective_function(self, TSAT_delay):
        """
        The objective function:
        Takes the TSAT_delay vector as input solution and returns 
        the chosen objective value
        """
        if self.optimizer =="DA" or "DE":
            # rounds to integer values:
            TSAT_delay = TSAT_delay.astype(int)
        #throughput, makespan, totalcost = self.evaluate(TSAT_delay)
        objective_value = self.evaluate(TSAT_delay)

        if self.objective == "throughput":
            return -objective_value  # To minimize
        
        if self.objective == "total_cost": 
            return objective_value # To minimize
        
        if self.objective == "makespan": 
            return objective_value # To minimize
        
        
    def optimize_TSATS(self):
        """Optimize the TSATs.
        
        Select Optimizer for TSAT optimization:
        Either:
        DE: Differential Evolution (scipy)
        DA: Dual Annealing (scipy)
        BF: Brute Force
        RS: Random Search
        TS: Tailored -  Single Greedy with No Local Search
        TSLS: Tailored as above - with Local Search
        TSSA: Tailored with Simulated Annealing
        TSLSSA: Tailored with LS and multiple SA
        TSLSSA: Tailored with LS and single SA
        TSRW: Tailord with Random Walk
        TSGD: Tailored with Greedy Descent

        """

        #Init the iter arrays with start at the FCFS solution for plots:
        self.iterations_best_throughput.append(self._FCFS_best_throughput)
        self.iterations_best_totalcost.append(self._FCFS_best_totalcost)
        self.iterations_best_makespan.append(self._FCFS_best_makespan)
        self._runtime.append(0) #Start time
        
        last_FCFS_TSAT = max(self._FCFS_TSATs)
        
        #TSAT constraints to enable any re-sequencing for the optimizers DA, DE and RS:
        bound_max = last_FCFS_TSAT + len(self._FCFS_TSATs)-1
        
        #Calculates the lower bounds for TSAT_delay vector based on TOBT and FCFS:
        self._TOBTs = np.array([flight['TOBT'] for flight in self.flight_schedule])
        lowerbound_TSAT_delay = self._TOBTs - self._FCFS_TSATs
        tsat_constraints = [(lowerbound_TSAT_delay[idx], bound_max) for idx in range(len(self._FCFS_TSATs))]
        
        # Initialize the starting point for TSAT delays at zero (gives FCFS)
        initial_solution = np.zeros(len(self._FCFS_TSATs))

        base_maxiter = 1000 # Default for now, for DE and DA.
        
        #Start runtime timing:
        self.start_time = time.time()

        if self.optimizer =="DE":
            # Scaling population size with dimensionality:
            scaled_maxiter = max(1, int(base_maxiter/self.dimensions))
            
            base_popsize = 10 #TBD Default for now.
            scaled_popsize = max(1, int(base_popsize * self.dimensions))  # Scale population size based on dimensions
            differential_evolution(self.objective_function, bounds = tsat_constraints, 
                                maxiter=scaled_maxiter, popsize=scaled_popsize,callback=self.callback, seed=self._optim_seed)
        elif self.optimizer =="DA":
            
            dual_annealing(self.objective_function, tsat_constraints, x0=initial_solution, maxiter=base_maxiter, callback=self.callback,seed=self._optim_seed)
        elif self.optimizer =="BF":
            self.brute_force_search(self.objective_function, self.dimensions, tsat_constraints, callback = self.callback)

        elif self.optimizer =="RS":
            self.random_search(self.objective_function, self.dimensions, tsat_constraints, callback = self.callback)
        
        elif self.optimizer =="TS":
            # TS Greedy allocation - No local search applied
            self.tailored_search(self.objective_function, local_search = False, callback = self.callback)

        elif self.optimizer =="TS_LS":
            # TS  + Local Search applied
            self.tailored_search(self.objective_function, local_search = True, callback = self.callback)

        elif self.optimizer =="TS_SA":
            # TS + Sim Anneal. No Local search
            if self._mode == "Det":
                target_accept_prob_init = 0.10
                #target_accept_prob_init = 0.80
                initial_temperature = 1
                
                #max_local_iter=10
                max_local_iter=900
            else: # For Noisy SA:
                target_accept_prob_init = 0.10
                initial_temperature = 1e-3
                max_local_iter=10
            
            self.simulated_annealing(self.objective_function, local_search=False, SA_start_iter=0, callback=self.callback, 
                            initial_temperature=initial_temperature, cooling_rate=0.95, iter_per_temp=1, 
                            max_global_iter=1000000, max_local_iter=max_local_iter, plotting=False,auto_temp=True,random_walk=False,greedy_descent=False,target_accept_prob_init=target_accept_prob_init,auto_cool=True)

        
        elif self.optimizer =="TS_LS_SA":
            # TS + LS + SA multiple runs
            if self._mode == "Det":
                target_accept_prob_init = 0.80
                initial_temperature = 1
                max_local_iter=10
                
            else:
                target_accept_prob_init = 0.1
                initial_temperature = 1e-3
                max_local_iter=10
            
            self.simulated_annealing(self.objective_function, local_search=True, SA_start_iter=0, callback=self.callback, 
                            initial_temperature=initial_temperature, cooling_rate=0.95, iter_per_temp=1, 
                            max_global_iter=1000000, max_local_iter=max_local_iter, plotting=False,auto_temp=True,random_walk=False,greedy_descent=False, target_accept_prob_init=target_accept_prob_init,auto_cool=True)
              
            
        elif self.optimizer =="TS_LS_SA_2":
            # TS + LS + SA single run
            if self._mode == "Det":
                target_accept_prob_init = 0.99
                initial_temperature =1e3
                max_local_iter=int(1e6)
            else:
                target_accept_prob_init = 0.90
                initial_temperature =1e2
                max_local_iter=int(1e6)

            self.simulated_annealing(self.objective_function, local_search=False, SA_start_iter=0, callback=self.callback, 
                            initial_temperature=initial_temperature, cooling_rate=0.95, iter_per_temp=1, 
                            max_global_iter=1000000, max_local_iter=max_local_iter, plotting=False,auto_temp=True,random_walk=False,greedy_descent=False,target_accept_prob_init=target_accept_prob_init,auto_cool=True)
             
           

        elif self.optimizer =="TSRW":
            # Random Walk with restarts by forcing acceptance to 1
            
            self.simulated_annealing(self.objective_function, local_search=False, SA_start_iter=0, callback=self.callback, 
                            initial_temperature=1e2, cooling_rate=0.95, iter_per_temp=100, 
                            max_global_iter=1000000, max_local_iter=900, plotting=False,auto_temp=False,random_walk=True,greedy_descent=False)
             
        elif self.optimizer =="TSGD":
            # Greedy Descent with restarts by forcing acceptance to 0
            
            self.simulated_annealing(self.objective_function, local_search=False, SA_start_iter=0, callback=self.callback, 
                            initial_temperature=1e2, cooling_rate=0.95, iter_per_temp=100, 
                            max_global_iter=1000000, max_local_iter=900, plotting=False,auto_temp=False,random_walk=False,greedy_descent=True)
             
        
        else:
            raise ValueError("Optimizer is not defined")
        

        best_TSAT = self.current_best_TSAT
        if self.printout:
            print("best TSAT")
            print(best_TSAT)

        optimized_objective = self._best_objective

        return best_TSAT, optimized_objective 
    

    
    def buffer_optimization_fixed_minutes(self):
        """Optimize buffer with increments based on the last move."""
        
        # Adjust the buffer based on the direction of the last move
        if self._last_buffer_move_improved:
            # Continue in the same direction if the last move improved
            
            if self.last_move_direction == "up":
                # Move towards the upper bound, shrinking the lower bound
                new_buff_min = self._buffer_minutes+0.5
                self._buffermin_lower_bound = (9 * self._buffer_minutes + self._buffermin_lower_bound) / 10

            elif self.last_move_direction == "down":
                # Move towards the lower bound, shrinking the upper bound
                new_buff_min = self._buffer_minutes-0.5
                self._buffermin_upper_bound = (new_buff_min + 9 * self._buffermin_upper_bound) / 10

            else:
                # If no previous direction, init with a small step up
                new_buff_min = self._buffer_minutes+0.2
                self.last_move_direction = "up"  # Default to increasing on first improvement
        else:
            # Switch direction if the last move did not improve
            if self.last_move_direction == "up":
                # Switch to moving downwards
                new_buff_min = self._buffer_minutes-0.2
                self._buffermin_upper_bound = (new_buff_min + 9 * self._buffermin_upper_bound) / 10
                self.last_move_direction = "down"
                
            elif self.last_move_direction == "down":
                # Switch to moving upwards
                new_buff_min = self._buffer_minutes+0.2
                self._buffermin_lower_bound = (9 * self._buffer_minutes + self._buffermin_lower_bound) / 10
                self.last_move_direction = "up"
                
            else:
                # If no direction, start by increasing the buffer
                new_buff_min = round((4*self._buffer_minutes + self._buffermin_upper_bound) / 5)
                new_buff_min = self._buffer_minutes+0.5
                self.last_move_direction = "up"

        self._buffer_minutes = new_buff_min
        self._buffer_list.append(self._buffer_minutes)
        
        # Update the range for buffer moves (diff between bounds)
        self.buffer_bound_range = self._buffermin_upper_bound - self._buffermin_lower_bound

    
        if self.printout:
            print(f"Updated buffer to {self._buffer_minutes} minutes, Last move direction: {self.last_move_direction}")
            print(f"Buffer bounds: {self._buffermin_lower_bound}, {self._buffermin_upper_bound}")
            print(f"Bound interval: {self.buffer_bound_range} minutes")
    
    
    def buffer_optimization_by_prob(self):
        """Optimize buffer based on last move and buffer optimization probability."""
        
        # Adjust the buffer based on the direction of the last move
        if self._last_buffer_move_improved:
            # Continue in the same direction if the last move improved
            if self.last_move_direction == "up":
                # Move towards the upper bound, shrinking the lower bound
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_upper_bound) / 10
                
                self.bufferprob_lower_bound =  (9 * self._bufferprob + self.bufferprob_lower_bound) / 10
                
        
            elif self.last_move_direction == "down":
                # Move towards the lower bound, shrinking the upper bound
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_lower_bound) / 10
                self.bufferprob_upper_bound = (new_buff_prob + 9 * self.bufferprob_upper_bound) / 10
            else:
                # If no previous direction, take a small step up
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_upper_bound) / 10
                self.last_move_direction = "up"  # Default to increasing on first improvement
        else:
            # Switch direction if the last move did not improve
            if self.last_move_direction == "up":
                # Switch to moving downwards
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_lower_bound) / 10
                self.bufferprob_upper_bound = (new_buff_prob + 9 * self.bufferprob_upper_bound) / 10

                self.last_move_direction = "down"
            elif self.last_move_direction == "down":
                # Switch to moving upwards
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_upper_bound) / 10
                self.last_move_direction = "up"
                self.bufferprob_lower_bound =  (9 * self._bufferprob + self.bufferprob_lower_bound) / 10
                
            else:
                # If no direction, start by incr the buffer
                new_buff_prob = (9 * self._bufferprob + self.bufferprob_upper_bound) / 10
                self.last_move_direction = "up"

        # Update buffer probability
        self._bufferprob = new_buff_prob
        
        self._buffer_list.append(self._bufferprob)
        
        # Update the range for buffer moves (diff between bounds)
        self.buffer_bound_range = self.bufferprob_upper_bound - self.bufferprob_lower_bound
        
        
        if self.printout:
            print(f"Updated buffer probability to {self._bufferprob}, Last move direction: {self.last_move_direction}")
            print(f"Buffer bounds: {self.bufferprob_lower_bound}, {self.bufferprob_upper_bound}")
            print(f"Bound interval: {self.buffer_bound_range}")


    def lower_bound_total_cost(self):
        """Takes the naive bound of the runway throughput cost
        and assumes all delay and wait cost could be avoided."""
        
        self.lowbound_cost = (self._lower_bound_makespan)*self.throughput_cost
        
        if self.printout:
            print(self.lowbound_cost)
        return self.lowbound_cost

        
    def min_positive_taxi_time(self):
        # Calculate the minimum positive value, returning 0 if all are zero
        # Generate non-zero values
        non_zero_values = (flight['exp_taxi_from_deice'] for flight in self.flight_schedule if flight['exp_taxi_from_deice'] > 0)
        self._min_pos_taxi = min(non_zero_values, default=0)
       
    def upper_bound_rwy_throughput(self):
        """Computes a upper_bound, the most restrictive of either 
        separation between flights (MILP solve) 
        or deice work load plus taxi times"""
        self.min_positive_taxi_time() 
        """Bound based on deice:"""

        self.lowbound_ADIT_serve_time = low_bound_deice_with_sep(self.flight_schedule, self.deicing_servers)
    
        self.earliest_all_deiced = self.lowest_ERZT_for_deice[-1] + self.lowbound_ADIT_serve_time
        
        self.earliest_all_takeoff_after_ice = self.earliest_all_deiced + self._min_pos_taxi
        if self.printout:
            print(self.earliest_all_takeoff_after_ice)
            
        
        """Bound based on separation"""
        
        self.total_separation_time = milp_separation_solver(self.flight_schedule)
        if self.printout:
            print("sep time:")
            print(self.total_separation_time)
            
        self.earliest_all_takeoff_for_separation = self.earliest_at_rwy + self.total_separation_time
        
        """Find the lower bound by either deice times, separation or latest TOBT time:"""

        if (
            self.earliest_all_takeoff_after_ice > self.earliest_all_takeoff_for_separation 
            and self.earliest_all_takeoff_after_ice > self.latest_at_rwy
        ):
            self._bounded_by = "Deice"
        elif (
            self.earliest_all_takeoff_for_separation > self.earliest_all_takeoff_after_ice 
            and self.earliest_all_takeoff_for_separation > self.latest_at_rwy
        ):
            self._bounded_by = "Separation"
        else:
            self._bounded_by = "Latest TOBT time"

        self._lower_bound_makespan = max(self.latest_at_rwy, self.earliest_all_takeoff_after_ice,self.earliest_all_takeoff_for_separation)
        print("Lower bound deterministic makespan is:")
        print(self._lower_bound_makespan)

        self._throughput_bound = self.number_flights*60/self._lower_bound_makespan
        if self.printout:
            print(f"Upper bound for RWY throughput per hour is {self._throughput_bound:.4} restricted by {self._bounded_by}.")
 
    

    def get_sorted_tsat_values(self, flights_data):
        sorted_tsat_values = [
            [flight['TSAT'] for flight in sorted(flight_list, key=lambda x: x['index'])]
            for flight_list in flights_data
        ]
        return sorted_tsat_values
    

    
    def callback(self,xk, f=None, context=None, convergence=None):
        """Callback function to terminate optimizers at bound or runtime elapsed"""
        
        run_time = time.time() - self.start_time
       
        if f is None:
            f = self.objective_function(xk)
        
        if self.objective =="throughput":
            bound_value = - self._throughput_bound
        
        if self.objective =="total_cost":
            bound_value = self.lowbound_cost

        if self.objective =="makespan":
            bound_value = self._lower_bound_makespan
        
        #Store the bound value:
        self._bound_value = bound_value

        # Check if the current solution meets the bound and we have stop at bound imposed:
        if f <= bound_value and self._stop_at_bound:
            
            f=float(f)
            bound_value = float(bound_value)

            print(f"Stopping optimizer {self.optimizer} since the {self.objective} of {-f:.2} has reached the bound of {-bound_value:.2}.")
            print(f"{self._num_elite_solutions} elite solutions and {self._num_rejected_solutions} rejects")

            #print(f"Callback function requested to stop early at global iteration {global_iterations} and local iteration {local_iterations} and total iteration {total_iterations}")
            print(f"Total Monte Carlo calls: {self._num_Cx_MC_calls} which made {self._num_Cx_calls} simulation calls")
            print(f" Duplicate runs: {self._num_duplicate_solutions}")
            
            return True
        
        if run_time > self._max_optimizer_time:
            print(f"Stopping optimizer {self.optimizer}. Runtime {run_time:.2f} seconds has exceeded the limit of {self._max_optimizer_time} seconds.")
            print(f"{self._num_elite_solutions} elite solutions and {self._num_rejected_solutions} rejects")
            print(f"Total Monte Carlo calls: {self._num_Cx_MC_calls} which made {self._num_Cx_calls} simulation calls")
            print(f" Duplicate runs: {self._num_duplicate_solutions}")
            
            return True
        
        if self.printout:
            print(f"Current solution: {xk}, Objective value: {f}, Runtime: {run_time:.2f} seconds")
        return False 



# Properties that are acessed below:
    
    @property
    def best_TSAT(self):
        return self.current_best_TSAT
    
    @property
    def best_throughput(self):
        return self._best_TSAT_throughput
    
    @property
    def best_makespan(self):
        return self._best_TSAT_makespan
    
    @property
    def best_totalcost(self):
        return self._best_TSAT_totalcost


    # Plots and Reports in latex and PDF:

    def plot_results(self, total_iterations, best_values, temperatures, accepted_values):
        """Plots the SA runs with the best values, accepted solutions, and temperature over total iterations"""
        
        fig, ax1 = plt.subplots()

        # Plotting best values on primary y-axis
        ax1.set_xlabel('Total Iteration')
        ax1.set_ylabel('Best Value', color='tab:blue')
        ax1.plot(total_iterations, best_values, color='tab:blue', label='Best Value')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Plotting accepted solutions on the same y-axis
        ax1.plot(total_iterations, accepted_values, color='tab:green', label='Accepted Value')
        # Create a second y-axis for temperature
        ax2 = ax1.twinx()
        ax2.set_ylabel('Temperature', color='tab:red')
        ax2.plot(total_iterations, temperatures, color='tab:red', label='Temperature')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Title and show the plot
        plt.title('Simulated Annealing: Value, Accepted Solutions, and Temperature Over Total Iterations')
        fig.tight_layout()
        
        plotname = "sim_anneal_demo"
        plt.savefig(plotname)
      
    

    def plot_combined_results(self, plotname="no_name", plot_bounds=True, plot_FCFS=True):
        """
        Generate combined plots from the results of multiple optimizers.
        """
        plt.figure(figsize=(10, 6))
        
        obj = self.objective
        results = self.results

        #plot bounds and FCFS:
        if obj == "total_cost":
            if plot_FCFS:
                plt.axhline(y=self._FCFS_best_totalcost, color='r', linestyle='--', label=f'FCFS total cost {self._FCFS_best_totalcost:.4}')
             

            if plot_bounds:
                    
                    plt.axhline(y=self.lowbound_cost, color='g', linestyle='--', label=f"Lower deterministic bound total cost {self.lowbound_cost:.3e}")

        elif obj == "throughput":
            if plot_FCFS:
                plt.axhline(y=self._FCFS_best_throughput, color='r', linestyle='--', label=f'FCFS throughput {self._FCFS_best_throughput:.4}')
             
            if plot_bounds:
                plt.axhline(y=self._throughput_bound, color='g', linestyle='--', label=f"Upper deterministic bound throughput {self._throughput_bound:.4}")
        
        elif obj == "makespan":
            if plot_FCFS:
                plt.axhline(y=self._FCFS_best_makespan, color='r', linestyle='--', label=f'FCFS makespan {self._FCFS_best_makespan}')
             
            if plot_bounds:
                plt.axhline(y=self._lower_bound_makespan, color='g', linestyle='--', label=f"Lower deterministic bound makespan {self._lower_bound_makespan}")
        
        
        # Line styles for plots:
        line_styles = ['-', '--', '-.', ':']

        # Iterator for line styles
        style_iterator = itertools.cycle(line_styles)

        # Iterate plots for each optimizer:
        for optimizername, data in results.items():
            # Get a unique line style and marker for each optimizer
            linestyle = next(style_iterator)

            if obj == "total_cost":
                plt.plot(data["runtime"], data["iterations_best_totalcost"],
                        linestyle=linestyle,
                        label=f"OPT: {optimizername}. Total cost {data['iterations_best_totalcost'][-1]:.4}")

            elif obj == "throughput":
                plt.plot(data["runtime"], data["iterations_best_throughput"],
                        linestyle=linestyle,
                        label=f"OPT: {optimizername}. Throughput {data['iterations_best_throughput'][-1]:.4}")

            elif obj == "makespan":
                plt.plot(data["runtime"], data["iterations_best_makespan"],
                        linestyle=linestyle,
                        label=f"OPT: {optimizername}. Makespan {data['iterations_best_makespan'][-1]}")
                

        plt.title(f'{plotname} Optimized {obj} vs Runtime')
        plt.xlabel('Runtime (seconds)')
        plt.ylabel(f'{obj} objective value')
        plt.grid(True)
        plt.minorticks_on()

        plt.xscale('log')  # Logarithmic scale on x-axis
        plt.yscale('log')  # Logarithmic scale on y-axis

        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')
        
        
        #else:
        plt.legend(loc='best', facecolor='white', framealpha=0.8)

        plt.savefig(plotname)
        plt.show()
   


    def save_results_report(self, report_name="no_name",short_summary = "None"):
        """
        Saves the results of the optimizers to three separate LaTeX tables:
        1. Objective Values, Gaps, and Performance Metrics
        2. Optimized TSAT and Flight Departure Sequence
        3. ADIT per pad, and Optimized Padlist
        """

        results = self.results

        report_data_obj = []
        report_data_opt_tsat = []
        report_data_adit = []

        # Summary of flight schedule:
        flights_sched = self.flight_schedule
        df = pd.DataFrame(flights_sched)
        sel_col = ['flight_id', 'aircraft_cat', 'SID','TOBT','deice_expected_ADIT']
        summary_flt_sched =df[sel_col]
        # Rename columns using a dictionary
        summary_flt_sched = summary_flt_sched.rename(columns={
            'flight_id': 'ID',
            'aircraft_cat': 'Cat',
            'deice_expected_ADIT': 'Expected ADIT'
        })

        latex_summary_flts = summary_flt_sched.to_latex(index=False, escape=False, caption=f'Flight Schedule {report_name}', label=f'tab:flt_sched_{report_name}', float_format="%.2f")
        # Replace \begin{table} with \begin{table}[htbp]
        latex_summary_flts = latex_summary_flts.replace(r"\begin{table}", r"\begin{table}[htbp]\\ \centering")


        for name, result in results.items():
            # First table: Objective Values, Gaps, Performance Metrics
            report_data_obj.append({
                'Optim': name,
                'OPT': result['optimized_objective'],
                'FCFS Gap': result["FCFS_gap"],
                '%FCFS': result["%\\imp"],
                'Bound Gap': result["Bound_gap"],
                '%GAP': result["%\\gap"],
                'Time to best (s)': result['best_runtime'],
                'Func Iter': len(result['runtime']) - 1,
            })

            # Second table: Optimized TSAT and Flight Departure Sequence
            def format_alignedarray(arr):
                return r'\alignedarray{' + ' & '.join(map(str, arr)) + '}'

            report_data_opt_tsat.append({
                'Optim': name,
                'OPTIMIZER TSATs': format_alignedarray(result['best_TSAT']),
                'Flight ID TSAT Sequence': format_alignedarray(result['offblock_seq']),
            })

            # Third table: Total ADIT, ADIT per pad, Optimized Padlist
            if result['ADIT_per_pad'] is not None:
                report_data_adit.append({
                    'Opt': name,
                    'ADIT per pad': result['ADIT_per_pad'],
                    'OPT Partition Flts to Pads': result['best_padlist'],
                })

        # Convert to DataFrames
        report_df_obj = pd.DataFrame(report_data_obj)
        report_df_opt_tsat = pd.DataFrame(report_data_opt_tsat)
        report_df_adit = pd.DataFrame(report_data_adit)

        # LaTeX Summary Notes with textwrap.dedent to remove indentation

        # For Summary report:
        fcfs_tsat = self._FCFS_TSATs
        fcfs_objective = self._FCFS_objective_val
        bound_objective = self._bound_value
        num_flights = self.number_flights
        num_servers = self.deicing_servers
        total_adit = self.deice_workload
        objective = self.objective
        max_runtime = self._max_optimizer_time
        best_optim = self.best_optimizer[0]
        best_value = self._best_value
        best_runtime = self.best_optimizer[1]['best_runtime']
        
        summary_latex = textwrap.dedent(fr"""\section*{{Results for: {report_name}}}
        \textbf{{Objective:}} {objective}.
        \textbf{{Max Runtime per Optimizer:}} {max_runtime} sec. \\
        \textbf{{Summary:}} {short_summary} \\
        \textbf{{Flights (n)}} = {num_flights}
        \textbf{{Deicing Servers (k)}}= {num_servers}
        \textbf{{Total ADIT}} = {total_adit} \\
        FCFS Objective = {fcfs_objective:.1f}, Bound = {bound_objective:.1f} \\
        \textbf{{Best Optimizer:}} {best_optim} with Objective Value = {best_value:.1f} and runtime: {best_runtime:.4f} sec.
        """)
        
        # Table 01:
        latex_header_obj = r"""
        \begin{table}[htbp]
        \centering
        \caption{Optimization Results}
        \label{{tab:optim_results}}
        \begin{tabular}{
        l
        S[table-format=2.2]
        S[table-format=1.2]
        S[table-format=2.2]
        S[table-format=1.2]
        S[table-format=1.2]
        S[table-format=1.2]
        S[table-format=4]
        }
        \toprule
        \textbf{Optimizer} & {\textbf{OPT}} & {\textbf{FCFS Gap}} & {\textbf{\% Imp}} & {\textbf{Bound Gap}} & {\textbf{\% Gap}} & {\textbf{Time to best (s)}} & {\textbf{Iter}}\\

        \midrule
        """

        # Table 02:
        latex_header_opt_tsat = r"""
        \begin{table}[htbp]
        \centering
        \caption{OPT TSAT and Flight ID TSAT Sequence}
        \label{tab:opt_tsat_flt_offblock_seq}
        \begin{tabular}{p{2.3cm} p{7cm} p{6cm}}
        \toprule
        \textbf{Optimizer} & \textbf{OPT TSATs} & \textbf{Flight ID TSAT Sequence} \\
        \midrule

       
        """
        # Table 03 ADIT partition:
        latex_header_adit = r"""
        \begin{table}[htbp]
        \centering
        \caption{ADIT and Padlist}
        \label{tab:adit_padlist}
        \begin{tabular}{llll}
        \toprule
        \textbf{Optimizer} & \textbf{ADIT per pad} & \textbf{OPT Partition Flts to Pads} \\
        \midrule

        """

        latex_footer = r"""
        \bottomrule
        \end{tabular}
        \end{table}
        """
        

        # Generate LaTeX content for each table
        latex_table_obj = report_df_obj.to_latex(index=False, header=False, float_format="%.2f", escape=False)
        
        # Remove unwanted LaTeX code
        lines = latex_table_obj.splitlines()
        lines_to_skip = ['\\begin{tabular}', '\\end{tabular}', '\\toprule', '\\midrule', '\\bottomrule']
        filtered_lines = [line for line in lines if not any(skip_line in line for skip_line in lines_to_skip)]
        # Join the remaining lines to form the table rows
        latex_table_obj = '\n'.join(filtered_lines)
        #print(latex_table_obj)
       
        # For report_df_opt_tsat and report_df_adit
        latex_table_opt_tsat = report_df_opt_tsat.to_latex(index=False, header=False, escape=False)

        # Remove unwanted LaTeX code
        lines = latex_table_opt_tsat.splitlines()
        lines_to_skip = ['\\begin{tabular}', '\\end{tabular}', '\\toprule', '\\midrule', '\\bottomrule']
        filtered_lines = [line for line in lines if not any(skip_line in line for skip_line in lines_to_skip)]
        # Join the remaining lines to form the table rows
        latex_table_opt_tsat = '\n'.join(filtered_lines)
        #print(latex_table_opt_tsat)

        latex_table_adit = report_df_adit.to_latex(index=False, header=False, escape=False)

        # Remove unwanted LaTeX code
        lines = latex_table_adit.splitlines()
        lines_to_skip = ['\\begin{tabular}', '\\end{tabular}', '\\toprule', '\\midrule', '\\bottomrule']
        filtered_lines = [line for line in lines if not any(skip_line in line for skip_line in lines_to_skip)]
        # Join the remaining lines to form the table rows

        # Modify latex_table_adit with \multirow and vertical stacking
        # Modify latex_table_adit with \multirow and vertical stacking
        # Dynamically generate latex_table_adit with \multirow and varying pad counts
        # Generate latex_table_adit with \multirow, dynamic pad count, and separation lines
        latex_table_adit = '\n'.join(
            [
                (
                    f"\\multirow{{{len(row['ADIT per pad'])}}}{{*}}{{{row['Opt']}}} & {row['ADIT per pad'][i]} & {row['OPT Partition Flts to Pads'][i]} \\\\"
                    if i == 0
                    else f"& {row['ADIT per pad'][i]} & {row['OPT Partition Flts to Pads'][i]} \\\\"
                )
                + ("\n\\midrule" if i == len(row['ADIT per pad']) - 1 else "")
                for _, row in report_df_adit.iterrows()
                for i in range(len(row['ADIT per pad']))
            ]
        )

        # Wrap latex_table_adit for final formatting
        latex_table_adit = f"""
            \\centering
            \\footnotesize
            {latex_table_adit}
        """


        # Combine each table's header, content, and footer
        full_latex_table_obj = latex_header_obj + latex_table_obj + latex_footer

        full_latex_table_opt_tsat = latex_header_opt_tsat + latex_table_opt_tsat + latex_footer
        full_latex_table_adit = latex_header_adit + latex_table_adit + latex_footer

        # Full LaTeX document content
        latex_preamble = textwrap.dedent(r"""
        \documentclass{article}
        \usepackage{booktabs}
        \usepackage{multirow}
        \usepackage{siunitx}
        \usepackage{array}
        \usepackage{geometry}
        \geometry{margin=1in}
        \usepackage[table]{xcolor}

        \usepackage{xparse}

        \NewDocumentCommand{\alignedarray}{O{10} m}{%
        \begin{tabular}{@{}*{#1}{>{\centering\arraybackslash}p{0.22cm}}@{}}%
        #2%
        \end{tabular}%
        }

        \newcommand{\formatarray}[1]{%
        [\begingroup\spaceskip=8pt\relax#1\endgroup]
        }

        \begin{document}
        """)

        latex_footer_doc = r"\end{document}"

        # Combine the LaTeX parts
        full_latex_code = (
            #latex_preamble
            summary_latex
            +latex_summary_flts
            +full_latex_table_obj
            + full_latex_table_opt_tsat
            + full_latex_table_adit
            #+ latex_footer_doc
        )

        # Save to LaTeX file
        with open(report_name + '.tex', 'w') as f:
            f.write(full_latex_code)

        print("LaTeX output with three tables and summary of this opt run is done")



    def header(self):
        self.set_font('Arial', 'B', 6)
        self.cell(0, 10, 'Results Report', ln=True, align='C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 6)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def save_results_report_pdf(self, report_name="no_name", short_summary="None", landscape_mode=True):
        results = self.results

        report_data_obj = []
        report_data_opt_tsat = []
        report_data_adit = []

        # Summary of flight schedule
        flights_sched = self.flight_schedule
        df = pd.DataFrame(flights_sched)
        sel_col = ['flight_id', 'aircraft_cat', 'SID','TOBT','deice_expected_ADIT']
        summary_flt_sched = df[sel_col].rename(columns={
            'flight_id': 'ID',
            'aircraft_cat': 'Cat',
            'deice_expected_ADIT': 'Expected ADIT'
        })

        summary_str = summary_flt_sched.to_string(index=False)

        for name, result in results.items():
            report_data_obj.append({
                'Optim': name,
                'OPT': result['optimized_objective'],
                'FCFS Gap': result["FCFS_gap"],
                '%FCFS': result["%\imp"],
                'Bound Gap': result["Bound_gap"],
                '%GAP': result["%\gap"],
                'Time to best (s)': result['best_runtime'],
                'Func Iter': len(result['runtime']) - 1,
            })

            report_data_opt_tsat.append({
                'Optim': name,
                'OPTIMIZER TSATs': ' '.join(map(str, result['best_TSAT'])),
                'Flight ID TSAT Sequence': ' '.join(map(str, result['offblock_seq'])),
            })

            if result['ADIT_per_pad'] is not None:
                report_data_adit.append({
                    'Opt': name,
                    'ADIT per pad': result['ADIT_per_pad'],
                    'OPT Partition Flts to Pads': result['best_padlist'],
                })

        report_df_obj = pd.DataFrame(report_data_obj)
        report_df_opt_tsat = pd.DataFrame(report_data_opt_tsat)
        report_df_adit = pd.DataFrame(report_data_adit)
        pdf = FPDF(orientation='L')
        #pdf = PDF(orientation='L' if landscape_mode else 'P')
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=6)

        # Summary Section
        pdf.cell(200, 10, txt=f"Results for: {report_name}", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(100, 5, txt=textwrap.dedent(f"""
            Objective: {self.objective}
            Max Runtime per Optimizer: {self._max_optimizer_time} sec
            Summary: {short_summary}
            Flights (n) = {self.number_flights}
            Deicing Servers (k) = {self.deicing_servers}
            Total ADIT = {self.deice_workload}
            FCFS Objective = {self._FCFS_objective_val:.1f}, Bound = {self._bound_value:.1f}
            Best Optimizer: {self.best_optimizer[0]} with Objective Value = {self._best_value:.1f} and runtime: {self.best_optimizer[1]['best_runtime']:.4f} sec
        """))
        pdf.ln(10)
        pdf.multi_cell(100, 10, txt="Flight Schedule Summary:")
        pdf.ln(5)
        pdf.set_font("Courier", size=9)  # Smaller font for fitting content
        pdf.multi_cell(100, 10, txt=summary_str)
        pdf.ln(10)

        # Objective Values Table
        pdf.set_font("Arial", 'B', 6)
        pdf.cell(0, 10, txt="Objective Values, Gaps, and Performance Metrics", ln=True)
        pdf.set_font("Arial", size=6)  # Adjust font size for large tables
        pdf.ln(5)
        for idx, row in report_df_obj.iterrows():
            row_text = ', '.join(f"{key}: {value}" for key, value in row.to_dict().items())
            pdf.multi_cell(100, 8, txt=row_text)
        pdf.ln(10)

        # Optimized TSAT Table
        pdf.set_font("Arial", 'B', 6)
        pdf.cell(100, 10, txt="Optimized TSAT and Flight Departure Sequence", ln=True)
        pdf.set_font("Arial", size=6)
        pdf.ln(5)
        for idx, row in report_df_opt_tsat.iterrows():
            row_text = ', '.join(f"{key}: {value}" for key, value in row.to_dict().items())
            pdf.multi_cell(100, 8, txt=row_text)
        pdf.ln(10)

        # ADIT and Padlist Table
        pdf.set_font("Arial", 'B', 6)
        pdf.cell(100, 10, txt="ADIT per Pad and Optimized Padlist", ln=True)
        pdf.set_font("Arial", size=6)
        pdf.ln(5)
        for idx, row in report_df_adit.iterrows():
            row_text = ', '.join(f"{key}: {value}" for key, value in row.to_dict().items())
            pdf.multi_cell(100, 8, txt=row_text)
        pdf.ln(10)

        pdf_filename = report_name + '.pdf'
        pdf.output(pdf_filename)
        print("PDF output with three tables and summary of this optimization run is done.")

"""version 105 Utils
Utility functions for this project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

import math
import random
import itertools
import copy #for deep copy
import heapq

import pulp #Only needed for the MILP solver.


def size_searchspace(padlist):
    """Input is list of flights divided into subsets.
    Finds the size of the search space when we only allow
    sequencing the flights within each subset and compares 
    this with the full search space sequencing flights"""
    combinations = 1 #Intitalise
    total_len = 0
    for pad in padlist:
        pad_len = len(pad)
        total_len +=pad_len
        padfac = math.factorial(pad_len)
        combinations = combinations*padfac
    total_fac = math.factorial(total_len)
    return combinations, total_fac

def sum_ones(A):
    """
    Counts the number of entries equal to 1 in matrix A.
    """
    counts = np.sum(A==1)
    return counts

def triangular_number(n):
    """
    Returns the sum of the n first natural numbers.
    """
    s =  n * (n + 1) // 2
    return s

def partition_with_diff(tot, parts, ones,lowest_ERZT_for_deice):
    """
    Partition the total workload into specified parts, considering 
    separations between flights and the earliest deicing times.

    Args:
        tot (float): The total workload to be divided.
        parts (int): The number of parts into which the workload will be partitioned.
        ones (int): The number of 1-minute separations between flights.
        lowest_ERZT_for_deice (list of float): A list of the earliest possible 
            times for deicing operations.

    Returns:
        tuple of numpy.ndarrays: 
            - x_array: The target partitioned workload vector across the deicing servers.
            - dep_deice_target: The adjusted departure time targets for deicing.
    
    """
    num_separations = parts-1
    n = max(0,num_separations - ones) #The number of two min separations in last flts.
    m = num_separations - n  # #The number of one min separations in last flights
    
    total_separation = triangular_number(m) + 2*triangular_number(n)
    
    delta_ERZT_times = [x - lowest_ERZT_for_deice[0] for x in lowest_ERZT_for_deice]

    diff_ERZT_add_time = sum(delta_ERZT_times)
    x_0 = max(0,(tot-total_separation+diff_ERZT_add_time)/parts) #to avoid x_0<0.
    s_ij = [1] * m + [2] * n #s_ij separation between flight i,j
    
    x_array = np.array([x_0 + sum(s_ij[:i]) for i in range(parts)])
    
    # Target vector for departure times from deice:
    dep_deice_target = x_array +lowest_ERZT_for_deice[0]
    
    # Target workload per server:
    x_array = x_array - delta_ERZT_times
    
    return x_array,dep_deice_target

def optimum_deice_array(flights,deice_workload, deicing_servers):
    """
    Generates the perfect partition of the deicing workload 
    to minimize total deicing time.

    Args:
    flights (list of dict): A list of flight data including 'expected_ERZT' and 'deice_expected_ADIT'.
    deice_workload (float): The total deicing workload.
    deicing_servers (int): The number of deicing servers available.

    Returns:
    numpy.ndarray: The partition target vector x_array for perfect deicing workload distribution, 
    and the dep_target numpy.ndarray which has the departures from deice with perfect partitioning.

    """
     # Find the lowest 'expected_ERZT' values that can go to deice first:
    lowest_ERZT_for_deice = heapq.nsmallest(deicing_servers, (flight['expected_ERZT'] for flight in flights))

    diff_ERZT_add_time = sum(x - lowest_ERZT_for_deice[0] for x in lowest_ERZT_for_deice[1:])
    
    sepmatrix = separation_matrix(flights)
   
    ones = sum_ones(sepmatrix)
    x_array,dep_target = partition_with_diff(deice_workload, deicing_servers, ones,lowest_ERZT_for_deice)
    
    return x_array,dep_target

def low_bound_deice_with_sep(flights, deicing_servers):
    """
    Computes the lower bound of the deicing makespan considering separations.

    Args:
    flights (list of dict): A list of flight data including 'expected_ERZT' 
    and 'deice_expected_ADIT'.

    deicing_servers (int): The number of deicing servers available.

    Returns:
    float or int: The lower bound of the deicing makespan.
    """

    deice_workload = sum(flight['deice_expected_ADIT'] for flight in flights)
   
    #Longest ADIT work on a single flight:
    max_workload = max(flight['deice_expected_ADIT'] for flight in flights)

    #Integer check in order to round up to integer for the makespan bound:
    all_integers = True
    for flight in flights:
        deice_time = flight['deice_expected_ADIT']
        integer_deice_time = deice_time ==int(deice_time)
        if not integer_deice_time:
            all_integers = False
    x_array,dep_target = optimum_deice_array(flights,deice_workload, deicing_servers)
    
    if all_integers:
        return max(max_workload, np.ceil(x_array[-1])) #lower bound makespan with integer
    else:
        
        return max(max_workload,x_array[-1]) #lower bound makespan deice times without integer


def greedy_partition(flights, target_array, deicing_servers):
    """
    The main algorithm for TS which allocates flights to deicing pads using a greedy algorithm to match the target workload distribution,
    while ensuring flights are not assigned to deice pads based on ADIT if it is critical for their ERZT timing.

    Args:
        flights (list of dict): A list of flight data, where each flight has 'deice_expected_ADIT' 
        (deicing service time) and 'earliest_ERZT', ready for deice time.
        target_array (numpy.ndarray): An array of target departures for each deicing pad.
        deicing_servers (int): The number of available deicing pads.

    Returns:
        tuple:
            sorted_deice_pads (list of lists): A list of lists, where each inner list represents the flights assigned to a deicing pad.
            sorted_deice_pad_totals (numpy.ndarray): The total deicing times for each deicing pad after allocation.
    """

    # Initialize deicing pads and their total deicing times
    deice_pads = [[] for _ in range(deicing_servers)]
    deice_pad_totals = np.zeros(deicing_servers)

    # Find the earliest ERZT values to start the pads
    flights_sorted_by_ERZT = sorted(flights, key=lambda x: x['earliest_ERZT'])
    
    # Init the starting times for each server (deicing pad) by choosing the k smallest ERZT values
    server_finish_times = [flights_sorted_by_ERZT[i]['earliest_ERZT'] for i in range(deicing_servers)]
    
    
    # While there are flights to assign:
    while (len(flights)>0):

        # Assigns flights that are ready (ERZT) to a deice server by greedy on ADIT:
        earliest_ready_server = min(server_finish_times)
        # Number of servers having the same unique early time ready:
        count_early_ready_servers = server_finish_times.count(earliest_ready_server)
        # Find flights ready:
        ready_flights = [flight for flight in flights if flight['earliest_ERZT'] <= earliest_ready_server]

        # If no flights are ready, adjust earliest server start time to match next ready:
        if len(ready_flights)==0:
            # Reset start time of earliest server to match the earliest flight
            first_ready_flight = min(flights, key=lambda x: x['earliest_ERZT'])
            
            sel_index = np.argmin(server_finish_times)
            server_finish_times[sel_index] = first_ready_flight['earliest_ERZT']
            continue

        #Sort ready flights by ADIT in descending order for greedy allocation
        ready_flights_sorted_by_ADIT = sorted(ready_flights, key=lambda x: x['deice_expected_ADIT'], reverse=True)
           
        # Greedy allocation of ready flights to deice pads
        for flight in ready_flights_sorted_by_ADIT:

            # Finds the diff from target:
            diff_array = target_array - server_finish_times
            
            # Decision on assign rule:
            if len(ready_flights_sorted_by_ADIT)>count_early_ready_servers:
                # ERZT not an issue, we can assign freely by ADIT:
                # Greedily assigns to the most underloaded of all server:
                sel_index = np.argmax(diff_array)

            else:
                # Assign by ADIT but only within subset of ERZT servers:
                # ADIT preference within the ERZT subset:
                min_indices = [index for index, fin_time in enumerate(server_finish_times) if fin_time == earliest_ready_server]
                  
                # Assigns to the most underloaded within this subset:
                sel_index = min_indices[np.argmax([diff_array[index] for index in min_indices])]

            # Update pads, pad totals, and server finish times:   
            deice_pads[sel_index].append(flight)
            deice_pad_totals[sel_index] += flight['deice_expected_ADIT']
            server_finish_times[sel_index] += flight['deice_expected_ADIT']
           
            # Remove the flight that has been assigned:
            flights.remove(flight)
            
    sorted_deice_pads, sorted_deice_pad_totals = deice_pads, deice_pad_totals
   
    # Sort the flights within pads by their earliest ERZT (ready for deice) times
    for pad in sorted_deice_pads:
        pad.sort(key=lambda x: x['earliest_ERZT'], reverse=False)
   
    # Update diff array:
    diff_array = target_array - server_finish_times

    return sorted_deice_pads, sorted_deice_pad_totals


def local_search(sorted_deice_pads, sorted_deice_pad_totals, target_array, lowbound = None, optim_seed=8, max_iter=500):
    """
    Local Search tries to improve the greedy allocation by moves in neighborhood.
    Takes the greedy solution and the target array as input

    The neigborhood N1, N2 and N3 moves are:
    1. N1: Moving flights with a rare SID direction to the end to reduce
    separation conflicts.
    2. N2 and N3: Probabilistically move (N3) or swap(N2) two flights between pads.

    Intention is to get a better partition, and as needed, to improve on separation
    conflicts between the last flights
    
    Args:
    sorted_deice_pads (list of lists): A list of lists where each inner list contains flights assigned to a deicing pad.
    sorted_deice_pad_totals (numpy.ndarray): The total deicing workload assigned to each deicing pad.
    target_array (numpy.ndarray): The target workload distribution for each deicing pad.
    lowbound (float, optional): A lower bound for the workload (if applicable). Default is None.
    optim_seed (int, optional): Random seed for reproducibility.
    max_iter (int, optional): Maximum number of iterations for local search. Default is 500.

    Returns:
    tuple:
        sorted_deice_pads (list of lists): A list of lists representing the updated flight assignments to deicing pads.
        sorted_deice_pad_totals (numpy.ndarray): The updated total deicing workloads for each deicing pad.

    Note: Local Search does not call the C(X) simulation.
    """

    num_deice_pads = len(sorted_deice_pad_totals)

    if num_deice_pads==1:
        raise ValueError("Local search cannot be used if only one deice pad")

    diff_array = target_array - sorted_deice_pad_totals

    # Neighbourhood_01 SID Swaps: 
    #Move rare SID flights, find the SID directions and counts:
    sid_directions, sorted_sid_counts = count_pads_sid_directions(sorted_deice_pads)
    most_freq_SID = next(iter(sorted_sid_counts))
    total_sid_count = np.sum(sid_directions, axis=0)
    most_freq_SID_count = np.max(total_sid_count)
    
    # Extract SID critical flights and move them to an optimum deice pad
    odd_index = [n for n in range(1,num_deice_pads) if n%2 !=0]
    odd_index_avail = odd_index.copy()
   
    moved_flights = []
    for padindex, pad in enumerate(sorted_deice_pads):
        for fltindex, flight in enumerate(pad):
            if flight['SID'] != most_freq_SID and flight not in moved_flights: #Detects a less freq SID
                adit_move = flight['deice_expected_ADIT'] #the ADIT moved
                index = padindex
                flt = fltindex
                
                moved_flights.append(flight)
                
                # Move this flight to a random pad with critical index
                crit_index = random.choice(odd_index_avail)
                
                sorted_deice_pads[index].remove(flight)
                sorted_deice_pads[crit_index].append(flight)
                
                #Removes the used index:
                odd_index_avail.remove(crit_index)
                if len(odd_index_avail) == 0:
                    #Restarts the index:
                    odd_index_avail = odd_index.copy()
                
    # Update after moving flights
    deice_pads = sorted_deice_pads
   
    pad_totals = sum_deice_pad_totals(deice_pads)

    counter = 1
    diff_array = target_array - pad_totals
    
    # Neighbourhood_02 and 03: Swaps or Moves: 
    # Begin local search iterations:
    while counter<max_iter:
        for i in range (len(pad_totals)):
            for j in range (len(pad_totals)):  
                counter +=1
                improved = False
                
                # Iterate over flights in deice pads i and j
                for idi, flight_i in enumerate(deice_pads[i]):
                    if flight_i in moved_flights:
                        if counter<100:
                            continue #skips critical flight
                    
                    for idj, flight_j in enumerate(deice_pads[j]):
                        if flight_j in moved_flights:
                            if counter<100:
                                continue #skips critical flight
                            
                        new_totals = pad_totals.copy()
                        
                        # Randomly select either a move or a swap:
                        swap_threshold = 0.9
                        swap_prob = random.random()

                        swap = (swap_prob>swap_threshold)
                        if swap: #Neighbourhood_02: Swap
                            #We perform a swap evaluation:
                            new_totals[i] = new_totals[i] - flight_i['deice_expected_ADIT'] + flight_j['deice_expected_ADIT']
                            new_totals[j] = new_totals[j] - flight_j['deice_expected_ADIT'] + flight_i['deice_expected_ADIT']
                        
                        else: #Neighbourhood_03: Move
                            #We only move flight from i to j evaluation:
                            new_totals[i] = new_totals[i] - flight_i['deice_expected_ADIT']
                            new_totals[j] = new_totals[j] + flight_i['deice_expected_ADIT']
                        
                        
                        current_deviation = np.max(pad_totals - target_array)
                        new_deviation = np.max(new_totals - target_array)

                        # Descent Algo: Only if the swap improves the deviation, perform the swap
                        # Does not evaluate C(X) directly, only f(x), eval of dev from target:
                        if new_deviation < current_deviation:
        
                            if swap:
                                
                                deice_pads[i].remove(flight_i)
                                deice_pads[j].remove(flight_j)
                                deice_pads[i].append(flight_j)
                                deice_pads[j].append(flight_i)
                            
                            else: #No swap, just move:
                            
                                deice_pads[i].remove(flight_i)
                                deice_pads[j].append(flight_i)

                            pad_totals = sum_deice_pad_totals(deice_pads)
                            diff_array = target_array - pad_totals

                            
                            improved = True
                            break

                    if improved:
                        break
        

    # Move all previously moved flights to the end of their assigned pad
    for pad in deice_pads:
        for flight in pad:
            if flight in moved_flights:
                pad.remove(flight)
                pad.append(flight)
    
    
    sorted_deice_pads, sorted_deice_pad_totals = deice_pads, pad_totals
    
   
    # Sort the flights within pads by their expected ERZT (ready for deice) times
    for pad in sorted_deice_pads:
        pad.sort(key=lambda x: x['expected_ERZT'], reverse=False)
  
    return sorted_deice_pads, sorted_deice_pad_totals
    

def last_flights(sorted_deice_pads):
    """returns the last flights from each sorted deice pad.
    It is used for evaluating the separation of the last flts"""
    last_flts = []
    for idx, flights in enumerate(sorted_deice_pads):
        if flights:  # Check if the pad has any flights
            last_flts.append(flights[-1])  # Access the last flight in the pad
    return last_flts

def separation_time(flights):
    """Separation time s_ij between a sequence of flights"""
    sep_time = np.zeros(len(flights)-1) #Between n-1 flts
    for idx, flight in enumerate(flights):
        if idx ==0:
            last_flight = flight
            continue #skip first flight

        current_cat = flight['aircraft_cat']
        prev_cat = last_flight['aircraft_cat']

        current_SID = flight['SID']
        prev_SID = last_flight['SID']

        if prev_cat == 'H' and current_cat in ['M', 'L']:
            sep_time[idx-1] = 2
    
        if prev_SID == current_SID:
            sep_time[idx-1] = 2
        else:
            sep_time[idx-1] = 1

        #set as last flight:
        last_flight = flight

    return sep_time


def sum_deice_pad_totals(deice_pads):
    """Sums the deice ADIT times S_i assigned to each deice_pad"""
    totals = np.zeros(len(deice_pads))
    for idx, deice_pad in enumerate(deice_pads):
        totals[idx] = np.sum([flight['deice_expected_ADIT'] for flight in deice_pad])
    return totals

def sort_deice_pads_by_total(deice_pads,deice_pad_totals):
    """Sorts the deice_pads by their ADIT sum total in ascending order"""
    pads_with_totals = list(zip(deice_pads,deice_pad_totals))
    sorted_pads_with_totals = sorted(pads_with_totals, key=lambda x: x[1])

    # Unzip the sorted pads and totals
    deice_pads_sorted, deice_pad_totals_sorted = zip(*sorted_pads_with_totals)
    deice_pads_sorted = list(deice_pads_sorted)
    deice_pad_totals_sorted = np.array(deice_pad_totals_sorted)

    return deice_pads_sorted, deice_pad_totals_sorted


def count_pads_sid_directions(deice_pads):
    """Counts the number of flights for each SID direction (N, E, S, W)
    in each deice pad and provides a total across all pads as well as a 
    dictionary of SIDs sorted by their total counts in descending order, 
    excluding SIDs with a count of zero."""
    
    sid_map = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
    sid_reverse_map = {0: 'N', 1: 'E', 2: 'S', 3: 'W'}

    # Initialize totals per pad
    sid_totals_per_pad = []

    # Iterate over each deice pad
    for deice_pad in deice_pads:
        # Initialize sid_counts for the current pad
        sid_counts = [0, 0, 0, 0]  # [N, E, S, W] counts for this pad
        
        # Count SID directions for each flight in the current pad
        for flight in deice_pad:
            sid = flight['SID']
            if sid in sid_map:
                idx = sid_map[sid]
                sid_counts[idx] += 1

        sid_totals_per_pad.append(sid_counts)
        
    # Convert list to a numpy array and sum elementwise to get totals across all pads
    sid_totals_per_pad = np.array(sid_totals_per_pad)
    sid_totals_all_pads = np.sum(sid_totals_per_pad, axis=0)

    # Create a dictionary of SIDs with their total counts, excluding zeros
    sid_totals_dict = {sid_reverse_map[idx]: count for idx, count in enumerate(sid_totals_all_pads) if count > 0}
    
    # Sort the dictionary by values (total counts) in descending order
    sorted_sid_totals_dict = dict(sorted(sid_totals_dict.items(), key=lambda item: item[1], reverse=True))
    
    return sid_totals_per_pad, sorted_sid_totals_dict


def deice_padlist(deice_pads):
    """Provides a nested list of flights assigned to each deice_pad"""
    deice_padlist = [[] for _ in range(len(deice_pads))] #empty nested list
    for index, deice_pad in enumerate(deice_pads):
        for flight in deice_pad:
            deice_padlist[index].append(flight['flight_id'])
    return deice_padlist
    

def add_noise(noise_value):
    
    """Returns uniform noise for EXOT noise from 0 to noise_value
    Note: The thesis assumes deterministic EXOT so this function is
    currently not in use."""
    return random.uniform(0,noise_value)


def fcfs_managed(flight_schedule, num_deice_pads, buffer):
    """
    Assigns TSATs for flights using an FCFS strategy, ensuring each flight is managed
    to arrive at an available deicing server, with a buffer added to the deicing arrival calculation.
    
    Parameters:
    - flight_schedule: List of flight dictionaries containing scheduling information.
    - num_deice_pads: Number of available deicing pads.
    - buffer: Fixed buffer time in minutes to ensure flights leave early to meet the deice server.
    
    Returns:
    - Updated flight_schedule with adjusted TSATs and timing information.
    """

    # Track of no deice flights:
    prev_nodeice_flight = None

    # Add 'original_index' to each flight
    flight_schedule = [{**flight, 'original_index': i} for i, flight in enumerate(flight_schedule)]

    # Create a list of server IDs (e.g., [0, 1, 2, ..., k-1] if there are k deicing pads)
    deicing_server_times = {server: 0 for server in range(num_deice_pads)}
    # List of previous flight for each server to keep track of ready times
    prev_flight_list = {prev_flight: None for prev_flight in range(num_deice_pads)}
    # Iterate over each flight in the order of their TOBT (already sorted)
    for flight in flight_schedule:
        # Init with values also for flights that do not need deice
        flight['TSAT'] = flight['TOBT']
        flight['expected_ERZT'] = flight['TSAT'] + flight['expected_EXOT']
        flight['ECZT'] = flight['expected_ERZT']

        if flight['deice_expected_ADIT'] > 0:
            
            sorted_servers = sorted(deicing_server_times.items(), key=lambda item: item[1])
            
            selected_server,avail_time = sorted_servers[0]
            prev_flt = prev_flight_list[selected_server]
           
            if prev_flt is None:
                        flight['expected_ERZT'] = flight['TSAT'] + flight['expected_EXOT']
                        flight['ECZT'] = flight['TSAT'] + flight['expected_EXOT']
            else:
                flight['expected_ERZT'] = max(flight['TSAT'] + flight['expected_EXOT'], prev_flt['expected_AEZT']- buffer)
                flight['ECZT'] = max(flight['TSAT'] + flight['expected_EXOT'], prev_flt['expected_AEZT'])

            # Adds this flight as the previous for this server:
            prev_flight_list[selected_server] = flight

            # Updates the selected deicing server's availability time for later sorting:
            deicing_server_times[selected_server] = flight['ECZT'] + flight['deice_expected_ADIT']

            # Calculate expected AEZT for all flights
            flight['expected_AEZT'] = flight['ECZT'] + flight['deice_expected_ADIT']
    
            #TSATS rounded to nearest minute:
            flight['TSAT'] = round(flight['expected_ERZT'] - flight['expected_EXOT'])
        
        else: # No deice:
            if prev_nodeice_flight is None:
                flight['TSAT'] = flight['TOBT']
            
            else:
                sepflights = prev_nodeice_flight,flight
                sep = int(separation_time(sepflights))
                flight['TSAT'] = prev_nodeice_flight['TSAT'] + sep
            
            prev_nodeice_flight = flight

    # Return the fcfc managed flight schedule:
    return flight_schedule


def generate_flight_schedule(num_flights, SOBT_start, SOBT_end, SID_dist, 
                             CAT_dist, expected_taxitime_range,
                             deice_service_rates, ice_need, cost_params, noise_params, 
                             instance_seed, five_min_interval=False):
    
    """
    Generates a flight schedule instance based on information available prior to departure from the gate.

    Parameters:
        num_flights (int): Number of flights to generate.
        SOBT_start (int): Start of Scheduled Off-Block Time (SOBT) range.
        SOBT_end (int): End of Scheduled Off-Block Time (SOBT) range.
        SID_dist (list): Weights for selecting SID (Standard Instrument Departure) directions.
        CAT_dist (list): Weights for selecting aircraft categories: L,M,H.
        expected_taxitime_range (tuple): Range (min, max) for EXOT expected taxi time from gate.
        deice_service_rates (tuple): Range (min, max) for deicing service times.
        ice_need (float): Probability that a flight needs deicing service.
        cost_params (tuple): Delay and wait cost per minute for medium and heavy aircraft.
        noise_params (dict): Noise parameters for EXOT, AOBT, and ADIT.
        instance_seed (int): Seed for random number generation.
        five_min_interval (bool, optional): Whether to generate SOBT times in 5-minute intervals.

    Returns:
        list: A list of dictionaries, each representing a flight with several attributes such as 
              SOBT, TOBT, EXOT, ADIT, deice service needs, noise and cost parameters.
              This list defines details that are known to the optimizers.
    """

    
    delay_cost_per_min_catM, wait_cost_per_min_catM, delay_cost_per_min_catH, wait_cost_per_min_catH = cost_params
    
    flight_schedule = []
    
    # Separate random number generator:
    rng = random.Random(instance_seed)
    
    # Use the separate RNG instance for generating random numbers
    random_list = [rng.randint(SOBT_start, SOBT_end) for _ in range(num_flights)]
    
    # Generate random times in 5-minute intervals
    if five_min_interval:
        random_list = [rng.randint(SOBT_start // 5, SOBT_end // 5) * 5 for _ in range(num_flights)]
    
    random_list.sort()  # Sorts the list in ascending order
    SOBT = random_list
    
    # Assign random Expected Taxitimes from gate
    expected_EXOT_min, expected_EXOT_max = expected_taxitime_range

    # Assign aircraft SID weighted on directions
    SID_directions = ['N', 'E', 'S', 'W']
    SID = rng.choices(SID_directions, weights=SID_dist, k=num_flights)

    # Assign Aircraft Category weighted on most Mediums
    Aircraft_Categories = ['H', 'M']
    acft_cat = rng.choices(Aircraft_Categories, weights=CAT_dist, k=num_flights)

    # Assign deice request and expected deice service times
    service_rate_low, service_rate_high = deice_service_rates
    deice_needs = [0, 1]

    # Stochastic noise
    exot_noise, aobt_noise, adit_sigma = (noise_params[key] 
                                          for key in ['exot_noise', 'aobt_noise', 'adit_sigma'])
    
    for i in range(num_flights):
        flight_id = f"Flight{i+1:02d}"
        exp_EXOT = rng.randint(expected_EXOT_min, expected_EXOT_max)  # Random Expected taxi time from gate to ice/rwy
        service_need = rng.choices(deice_needs, weights=[1 - ice_need, ice_need])[0]
        
        if service_need == 0:
            # Dummy values - no deice
            exp_ADIT = 0
            exp_ADIT_sigma = 0
            exp_ADIT_mu = 0
            exp_taxi_from_deice = 0
        else:
            exp_ADIT = rng.randint(service_rate_low, service_rate_high)
            exp_ADIT_sigma = adit_sigma  # sigma for lognormal
            exp_ADIT_mu = np.log(exp_ADIT) - (adit_sigma**2 / 2)
            exp_taxi_from_deice = 4  # Default for now


        if acft_cat[i] == 'M':
            wc = wait_cost_per_min_catM
            dc = delay_cost_per_min_catM
        else:
            wc = wait_cost_per_min_catH
            dc = delay_cost_per_min_catH

        flight_schedule.append({
            "flight_id": flight_id,
            "aircraft_cat": acft_cat[i],
            "SOBT": SOBT[i],
            "TOBT": SOBT[i],
            "SID": SID[i],
            "expected_AOBT_noise": aobt_noise,
            "expected_EXOT": exp_EXOT,
            "expected_EXOT_noise": exot_noise,
            "deice_service_need": service_need,
            "deice_expected_ADIT": exp_ADIT,
            "deice_expected_ADIT_sigma": exp_ADIT_sigma,
            "deice_expected_ADIT_mu": exp_ADIT_mu,
            "deice_expected_wait": 0,  # Initialize with 0 before optimization
            "buffer_EXOT": 0,  # Init at 0 min buffer
            "buffer_EXOT_prob": 0.85,  # Init at 0.85 prob, will be reset by optimizer
            "exp_taxi_from_deice": exp_taxi_from_deice,
            "wait_cost_per_min": wc,
            "delay_cost_per_min": dc
        })

    for flight in flight_schedule:
        flight['TSAT'] = flight['TOBT']  # Initialize TSATs as TOBTs
        # Estimated time ready for deice based on sum of expectations of TOBT plus taxi time
        flight['earliest_ERZT'] = flight['TOBT'] + flight['expected_EXOT']
        flight['expected_ERZT'] = flight['TOBT'] + flight['expected_EXOT']
        flight['expected_ECZT'] = flight['expected_ERZT'] + flight['deice_expected_wait']
        flight['expected_AEZT'] = flight['expected_ECZT'] + flight['deice_expected_ADIT']
        flight['expected_ETOT'] = flight['expected_AEZT'] + flight['exp_taxi_from_deice']

    return flight_schedule


def generate_flights(flight_schedule, mode= "Det", antithetic=False):
    """Takes the flight_schedule and adds randomness to each simulation run, 
    based on the expected values provided.
    Random deice service times added - based on the expected service time"""

    """
    Takes the flight_schedule and adds randomness to each simulation run, 
    based on the expected values provided. Random deice service times are added 
    based on the expected service time.

    Parameters:
        flight_schedule (list): List of dictionaries representing flights with attributes 
                                such as TOBT, expected EXOT, ADIT etc.
        mode (str, optional): Mode of operation, "Det" for deterministic or "Stochastic" for 
                              adding randomness. Defaults to "Det".
                              With "Det" all values are returned equal to the expectation.
                              With "Sto" noise is added.
        antithetic (bool, optional): Whether to use antithetic variates for generating noise. 
                                     Defaults to False.

    Returns:
        list: Realized flights with values based on the specified mode 
        and randomness.
    """


    
    num_flights = len(flight_schedule)
    flights = flight_schedule
    
    for flight in flights:
        aobt_noise, aobt_anti_noise = generate_uniform_sample_with_antithetic(-flight['expected_AOBT_noise'], flight['expected_AOBT_noise'])
        
        flight['ARDT'] = flight['TOBT'] + aobt_noise
        flight['EXOT'] = flight['expected_EXOT']
        #flight ['AOBT'] = flight ['TSAT'] #Will be replaced by stochastics in simulator.
        if mode == "Det":
            flight ['ARDT'] = flight ['TOBT']
            flight ['AOBT'] = flight ['ARDT']
            flight['ADIT'] = flight['deice_expected_ADIT']
            
        else: #Stochastic:
           
            #Actual deice ADIT time:
            mu = flight['deice_expected_ADIT_mu']
            
            if mu>0:
                
                flight['ADIT'] = generate_lognormal_sample(mu,flight['deice_expected_ADIT_sigma'])
                
            else: #For flights with no deice need:
                
                flight['ADIT'] = 0
            
    return flights


def generate_lognormal_sample(mu, sigma):
    """
    Generates a lognormal sample from the given mu and sigma.

    Parameters:
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.

    Returns:
        float: A lognormal sample
    """
    
    return np.random.lognormal(mu, sigma)

def generate_lognormal_sample_with_antithetic(mu, sigma):
    """
    Generates a lognormal sample from the given mu and sigma,
    and its antithetic counterpart.

    Parameters:
        mu (float): The mean of the underlying normal distribution.
        sigma (float): The standard deviation of the underlying normal distribution.

    Returns:
        tuple: A pair of lognormal samples that are antithetic.
    """

    z = np.random.normal()
    x = np.exp(mu + sigma * z)
    x_anti = np.exp(mu - sigma * z)
    return x, x_anti

def generate_uniform_sample_with_antithetic(a, b):
    """Returns a uniformly distributed U in the range (a, b) 
    and its antithetic U_anti"""
    U = random.uniform(a, b)
    U_anti = a + (b - U)
    return U, U_anti


def lognormal_survival_time(mu_i, sigma_i, prob):
    """ Returns the survival ADIT time for job i given lognormal service of job i and prob of not finished"""
    z = norm.ppf(1-prob)
    # Lognormal with given z
    survival_time = np.exp(mu_i + z * sigma_i)
    return survival_time
  

def generate_flights_anti(flight_schedule, mode="Det", antithetic=True):
    """
    Takes the flight_schedule and adds randomness to each simulation run, 
    based on the expected values provided. Random deice service times added 
    based on the expected service time.
    """

    num_flights = len(flight_schedule)
    flights = flight_schedule  # Regular flights
    flights_anti = copy.deepcopy(flight_schedule)  # Antithetic flights

    for flight, flight_anti in zip(flights, flights_anti):
        # Random AOBT and EXOT taxi out time:
        
        flight['AOBT'] = flight['TSAT']  # Will be replaced by stochastics in simulator
        flight_anti['AOBT'] = flight_anti['TSAT']  # Same for antithetic version

        if antithetic:
            aobt_noise, aobt_anti_noise = generate_uniform_sample_with_antithetic(-flight['expected_AOBT_noise'], flight['expected_AOBT_noise'])
           
            flight['ARDT'] = flight['TOBT'] + aobt_noise
            flight_anti['ARDT'] = flight_anti['TOBT'] + aobt_anti_noise
            

        if mode == "Det":
            raise ValueError("Cannot combine Antithetic with Deterministic mode")

        else:  # Stochastic mode
            exot_noise = add_noise(flight['expected_EXOT_noise'])
            flight['EXOT'] = flight['expected_EXOT'] + exot_noise

            exot_noise_anti = add_noise(flight_anti['expected_EXOT_noise'])
            flight_anti['EXOT'] = flight_anti['expected_EXOT'] + exot_noise_anti

            # Deice ADIT mu:
            mu = flight['deice_expected_ADIT_mu']
            mean = flight['deice_expected_ADIT']
            if mu > 0:
                if antithetic:
                    adit, adit_anti = generate_lognormal_sample_with_antithetic(mu, flight['deice_expected_ADIT_sigma'])
                    flight['ADIT'] = adit
                    flight_anti['ADIT'] = adit_anti
                    
                else:
                    flight['ADIT'] = generate_lognormal_sample(mu, flight['deice_expected_ADIT_sigma'])
                    flight_anti['ADIT'] = generate_lognormal_sample(mu, flight_anti['deice_expected_ADIT_sigma'])
            else:
                # For flights with no deice need:
                flight['ADIT'] = 0
                flight_anti['ADIT'] = 0

    if antithetic:
        return flights, flights_anti
    else:
        return flights



def policy_dict(Policy_ICE, max_wait_after_deice):
    """returns a dictionary of policies for 
    the flights. Currently the policy for
    resequencing flights in the simulation, 
    after deicing in order to be released for taxi
    to the runway.s
    
    This resequence is for SID or wake turb separation.
    
    Inputs:
    Policy_ICE:  is set to "RESEQ_ICE" or "FCFS"
    max_wait_after_deice (float) is set to minutes max wait.
    """
    
    policies = {
        "Policy_ICE": {
            "type": Policy_ICE,  # or "FCFS"
            #"type": "FCFS",
            "max_wait_after_deice": max_wait_after_deice  # max minutes wait and hold if RESEQ_ICE
        }
    }
    return policies


def separation_matrix(flights):
    """takes a flights info array and returns
    separation matrix for Vortex and SID sep"""
    
    N = len(flights)

    sep_mat = np.zeros((N, N), dtype=int)
    for f in range(N):
        for g in range(N):
            if f != g:
                sep = 1  # default
                if flights[g]['aircraft_cat'] in ['M', 'L'] and flights[f]['aircraft_cat'] == 'H':
                    sep = 2
                if flights[f]['SID'] == flights[g]['SID']:
                    sep = 2
                sep_mat[f, g] = sep
    
    return sep_mat

def milp_separation_solver(flight_schedule):
    """Only used to calculate a lower bound makespan for separation
    not considering any deice. In most practical cases the deicing LB
    is more limiting."""
    # Problem:
    prob = pulp.LpProblem("Flight_Sequence_Minimization", pulp.LpMinimize)

    # Number of flights
    n = len(flight_schedule)
    #The separation matrix (nxn) for the flights, which returns 1 or 2 in each cell:
    separation = separation_matrix(flight_schedule)

    # Decision variables
    x = pulp.LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n)), cat='Binary')
    #For speed we allow u to be continuous:
    u = pulp.LpVariable.dicts("u", (i for i in range(n)), lowBound=0, upBound=n-1, cat='Continuous')

    # Objective function: Minimize the total separation time
    prob += pulp.lpSum(separation[i][j] * x[i, j] for i in range(n) for j in range(n))

    # Constraints
    # Total sequence connections should be n-1
    prob += pulp.lpSum(x[i, j] for i in range(n) for j in range(n)) == n - 1

    # Each flight can have at most one successor
    for i in range(n):
        prob += pulp.lpSum(x[i, j] for j in range(n)) <= 1

    # Each flight can have at most one predecessor
    for j in range(n):
        prob += pulp.lpSum(x[i, j] for i in range(n)) <= 1

    # No self-loop
    for i in range(n):
        prob += x[i, i] == 0

    # Prevent cycles by MTZ subtour elimination:
    for i in range(1, n):
        for j in range(1, n):
            if i != j:
                prob += u[i] - u[j] + n * x[i, j] <= n - 1

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    total_sep = pulp.value(prob.objective)
   
    return total_sep

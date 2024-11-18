import itertools
import numpy as np
import random

from utils import optimum_deice_array,greedy_partition,local_search,deice_padlist,sort_deice_pads_by_total, size_searchspace


"""Own Tailored Sorting algorithm"""

class FlightOptimizer:
    def __init__(self, flights, num_deice_pads, local_search = True, optim_seed = 8):
        
        self._optim_seed = optim_seed
        self._rng_optim = np.random.default_rng(seed=self._optim_seed)
        
        # Add index to each flight
        self.flights = [{**flight, 'original_index': i} for i, flight in enumerate(flights)]
        self.num_deice_pads = num_deice_pads
        self.deice_workload = sum(flight['deice_expected_ADIT'] for flight in self.flights)

        #Target vectors:
        self._x_array, self._dep_array_target = optimum_deice_array(self.flights,self.deice_workload, self.num_deice_pads)
        
    
        self._local_search = local_search
        self.deice_pads_flights, self.deice_pad_totals = self.allocate_flights_to_deice_pads()
        
    def allocate_flights_to_deice_pads(self):
        
        #Greedy Solution:
        
        deice_pads,deice_pad_totals = greedy_partition(self.flights,self._dep_array_target,self.num_deice_pads)
        

        #print("Deice Set from Greedy:")
        padlist = deice_padlist(deice_pads)
        
        if self._local_search and self.num_deice_pads>1:
            print("Local search applied on the greedy solution:")
         
            deice_pads,deice_pad_totals = local_search(deice_pads,deice_pad_totals, self._x_array,self._optim_seed)
        
        departure_array = deice_pad_totals
        for i in range(1, len(departure_array)):
        # If the difference between the current and previous element is less than 1
            if departure_array[i] <= departure_array[i - 1] + 1:
            # Increment the current element to be exactly 1 greater than the previous element
                departure_array[i] = departure_array[i - 1] + 1
        
        diff_array = self._x_array - deice_pad_totals
        
        self.partition_loss = departure_array[-1] - self._x_array[-1] #Loss due to partition
       
        for index, deice_pad in enumerate(deice_pads):

             # Alternate sorting order based on the index of the deice pad
             reverse_order = (index % 2 == 1)  # True for odd indices
             deice_pad = sorted(deice_pad, key=lambda x: x['expected_ECZT'], reverse=False)
             if len(deice_pad) > 1:
                 #alternate sorting:
                 deice_pad[1:] = sorted(deice_pad[1:], key=lambda x: x['deice_expected_ADIT'], reverse=reverse_order)

        
        return deice_pads, deice_pad_totals


    def get_sorted_tsat_values(self, flights_data):
        sorted_tsat_values = [
            [flight['TSAT'] for flight in sorted(flight_list, key=lambda x: x['index'])]
            for flight_list in flights_data
        ]
        return sorted_tsat_values
    
 
    def generate_all_permutations(self):
        """Generates all permutations - generator function"""
        deice_pad_permutations = (itertools.permutations(deice_pad) for deice_pad in self.deice_pads_flights)
        
        # Generate Cartesian product across all deice pads
        all_combinations = itertools.product(*deice_pad_permutations)

        # Process combinations one by one and yield results
        for combination in all_combinations:
            #combined_tsats = []
            for perm in combination:
                perm_copy = [flight.copy() for flight in perm]
                
            yield perm_copy
    




    def generate_random_permutations(self, threshold=20000, max_sample_size=10000000):
        """Generate permutations. For search space below threshold size we
        generate all permutations in memory, for larger we generate random"""
        
        size, totalsize = size_searchspace(self.deice_pads_flights)
        
        print("The search space size is:", size)
        
        if size <= threshold:
            """Generate all permutations and cycle through them indefinitely"""
            # Generate all permutations for each deice pad
            deice_pad_permutations = (itertools.permutations(deice_pad) for deice_pad in self.deice_pads_flights)
            
            # Generate Cartesian product across all deice pads
            all_combinations = itertools.product(*deice_pad_permutations)

            # Use itertools.cycle to loop the permutations indefinitely
            infinite_combinations = itertools.cycle(all_combinations)

            # Cycle through all combinations indefinitely
            while True:
                combination = next(infinite_combinations)
                perm_copy = [[] for _ in range(len(self.deice_pads_flights))]
                for index, perm in enumerate(combination):
                    # Create a copy of the current permutation for the current deice pad
                    perm_copy[index] = [flight.copy() for flight in perm]
                
                yield perm_copy

        else:
            """Large search space - generate random samples (which may not be unique)"""
            
            for idx_search in range(max_sample_size):
                perm_copy = [[] for _ in range(len(self.deice_pads_flights))]  # list of permutation copy
                for index, deice_pad in enumerate(self.deice_pads_flights):
                    # Randomly shuffle the flights within each deice pad
                    shuffled_deice_pad = deice_pad[:]
                    self._rng_optim.shuffle(shuffled_deice_pad)
                    perm_copy[index] = [flight.copy() for flight in shuffled_deice_pad]

                yield perm_copy
          

            





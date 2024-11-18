"""version 105"""

import simpy
import time
import pandas as pd
from tabulate import tabulate 

import random
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps

#For optimizing:
from scipy.optimize import dual_annealing #Simulated Annealing.
from scipy.optimize import minimize
from scipy.optimize import differential_evolution #Basically a newer version of GA
import itertools #for brute force iterations.

#For QQ plots
import scipy.stats as stats

#Own modules:
from utils import generate_flight_schedule, generate_flights,generate_flights_anti

class FlightSimulation:
    """Simulates a two stage queue system with deicing and a single departure rwy
    printout=True provides print out of event log"""
    def __init__(self, flight_schedule, deicing_servers, throughput_cost, policies, printout=False, mode="Det",max_sim_time=1000, antithetic=False):
        
        #Run simulation as deterministic or stochastic:
        if mode== "Det" or mode=="Sto":
            self.mode = mode
            
        else:
            raise ValueError("Mode is not defined")
        
        # If MC simulation will run sets of antithetic samples in individual runs:
        self._antithetic = antithetic

        #The flight information known beforehand at Init:
        self.flight_schedule = flight_schedule
        self._SOBTs = np.array([flight['SOBT'] for flight in self.flight_schedule])
        self._TOBTs = np.array([flight['TOBT'] for flight in self.flight_schedule])
        self._expected_EXOTs = np.array([flight['expected_EXOT'] for flight in self.flight_schedule])
       
        self._deice_service_need = np.array([flight['deice_service_need'] for flight in self.flight_schedule]) #Deice need or not.
        self._taxi_time_from_deice = np.array([flight['exp_taxi_from_deice'] for flight in self.flight_schedule]) #Deterministic.
        self._flight_ids = [flight['flight_id'] for flight in self.flight_schedule]
        self._categories = [flight['aircraft_cat'] for flight in self.flight_schedule]
        self._SIDs = [flight['SID'] for flight in self.flight_schedule]
        
        #Cost Params b2 and b1 for curve fit:
        self._wait_cost_per_mins_b2_b1 = [flight['wait_cost_per_min'] for flight in self.flight_schedule]
        self._delay_cost_per_mins_b2_b1 = [flight['delay_cost_per_min'] for flight in self.flight_schedule]

        # Access of b2 and b1 coefficients:
        self._delay_b2 = [coeff[0] for coeff in self._delay_cost_per_mins_b2_b1]
        self._delay_b1 = [coeff[1] for coeff in self._delay_cost_per_mins_b2_b1]
        
        self._wait_b2 = [coeff[0] for coeff in self._wait_cost_per_mins_b2_b1]
        self._wait_b1 = [coeff[1] for coeff in self._wait_cost_per_mins_b2_b1]
       

        self.deicing_servers = deicing_servers
        
        self.printout = printout
        self.max_sim_time = max_sim_time

        self.num_flights = len(self.flight_schedule)

        #Setting up the Airport (A) simulation environment with resources:
        self.env = simpy.Environment()
        self.deicing_resource = simpy.Resource(self.env, capacity=deicing_servers)
        self.runway_resource = simpy.Resource(self.env, capacity=1) #Single dep runway.
        #Airport throughout cost:
        self.throughput_cost = throughput_cost

        #For book keeping of the deice system and time integrated metrics:
        self._AEZT = np.zeros(self.num_flights) #End of deice time
        self._ACZT = np.zeros(self.num_flights) #Start of deice time
        self._deice_deice_departure_order = None

        self._total_in_deice_system = 0 #Includes waiting in queue and being served.
        self._area_in_deice_system = 0
        self._area_in_deice_queue = 0
        self._time_in_deice_system = [(0, 0)]
        self._time_in_deice_queue = [(0, 0)]
        self.total_in_taxi_from_deice = 0

        self.deicing_users = {} #Keeps track of flights currently in deice service

        #For book keeping of the runway system
        self._RTOTs = np.zeros(self.num_flights) #Own definition; Ready for Take Off.
        self._ATOTs = np.zeros(self.num_flights)
        self.rwy_interdep_times = np.zeros(self.num_flights)
        self._rwy_truput_per_hr = np.zeros(self.num_flights)

        self._last_in_line_taxi_to_rwy = None #For sequencing logic.
        self._last_taxi_to_rwy_event_time = 0 #For sequencing logic.
        self._number_of_takeoffs = 0
        self._total_in_runway_system = 0
        self._area_in_runway_system = 0
        self._area_in_runway_queue = 0
        self._time_in_runway_system = [(0, 0)]
        self._time_in_runway_queue = [(0, 0)] #Time at holding point

        #Departures from the runway:
        self._last_takeoff_event_time = 0
        self._first_takeoff_event_time = 0 #Start time for runway metrics
        self._last_acft_cat = None
        self._last_SID = None
        
        #Optimization release for taxi from deice logic:
        self._acft_requesting_taxi_from_deice = [] #list of aircraft done deice and req taxi.
        self.interrupt_event = self.env.event()  # Event to interrupt the wait
        
        #Optimization Times:
        self._TSATs = np.array([flight['TSAT'] for flight in self.flight_schedule])
        
        self._TTOTs = np.zeros(self.num_flights) #Target Take Off Time
        self._ETOTs = np.zeros(self.num_flights) #Estimated take off time

        #Optimization Policies:
        self.policies = policies  # Store a policy dictionary
        
        #General book keeping:
        self.event_log = []
        self._last_event_time = 0

        #Performance book keeping:
        self.rwy_idle_time_loss = 0 #accumulator of idle time
        self._rwy_sep_time_loss = 0 #accumulator of sep time loss

        self._rwr_sep_loss_per_hr = np.zeros(self.num_flights) #Stores minutes of separation efficiency loss rwy
        self._rwy_idle_loss_per_hr = np.zeros(self.num_flights) #Stores minutes of lost rwy utilization due no flights

    def run(self):

        """Generates a set of random flights based on 
        init flight_schedule and the mode
        being either deterministic or stochastic"""
       
        if self._antithetic:
            #Note: Stochastic gen flights for x and x_anti is done in antithetic Monte Carlo,
            # So here the flight schedule is now stochastic:
            self.flights = self.flight_schedule
        
        else:
            # Generates a set of random flights:
            self.flights = generate_flights(self.flight_schedule,self.mode)
            
        
        if self.mode == "Det":
            self._AOBTs = self._TSATs
            
            
        else:
        # For stochastic mode the ready time ARDT:
            self._ARDTs = np.array([flight['ARDT'] for flight in self.flights])
            
            #Then set AOBT to be the maximum of the ARDT and the TSATs:
            # Note that TSATs are set by the optimizer.
            self._AOBTs = np.maximum(self._TSATs, self._ARDTs)
           
        self._EXOTs = np.array([flight['EXOT'] for flight in self.flights])
        self._ADITs = np.array([flight['ADIT'] for flight in self.flights])
        
        self._ERZTs = self._AOBTs + self._EXOTs
        
        """Set up and run the SimPy simulation for each flight."""
        for i in range(self.num_flights):
            self.env.process(self.complete_process(i))
        
        self.env.run(until = self.max_sim_time)

        # Create a sorted list of flight indices:
        sorted_indices = np.argsort(self._AEZT)
        self._deice_deice_departure_order = [self._flight_ids[i] for i in sorted_indices]

    
    def requires_run(method):
        """Ensure the simulation has been run before accessing properties."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            if self._deice_deice_departure_order is None:
                raise ValueError("The simulation has not been run yet. Please run the simulation first.")
            return method(self, *args, **kwargs)
        return wrapper

  
    def event_logging(self, index, event):
        """Creates a log dict of all events"""
        self.event_log.append({
            'Time': f'{self.env.now:.2f}',
            'FlightID': self._flight_ids[index],
            'Event': event,
            'Deice Sys': self._total_in_deice_system,
            'Deice Queue': self.total_in_deice_queue,
            'Total in Runway System': self._total_in_runway_system,
            'Runway Queue': self.total_in_runway_queue,
            'Taxiways from Deice': self.total_in_taxi_from_deice
        })
        

    def print_log(self):
        # Convert event log to DataFrame
        df = pd.DataFrame(self.event_log)
        # Print the DataFrame as a table using tabulate
        print("\nEvent Log:")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))


    def save_log_to_csv(self, filename):
        df = pd.DataFrame(self.event_log)
        df.to_csv(filename, index=False)

    def save_log_to_excel(self, filename):
        df = pd.DataFrame(self.event_log)
        df.to_excel(filename, index=False)

    def save_log_to_html(self, filename):
        df = pd.DataFrame(self.event_log)
        df.to_html(filename, index=False)

    def save_log_to_latex(self, filename):
        df = pd.DataFrame(self.event_log)
        with open(filename, 'w') as f:
            f.write(df.to_latex(index=False))


    @property
    def total_in_deice_queue(self):
        """Return the current number of flights in the deice queue."""
        return max(0, self._total_in_deice_system - self.deicing_resource.capacity)
        
    
    @property
    def total_in_runway_queue(self):
        """Return the current number of flights in the runway (holding) queue."""
        return max(0, self._total_in_runway_system - self.runway_resource.capacity)
        #return max(0, self._total_in_runway_system)
    
    @property 
    def sep_loss(self):
        """Return the current accumulated runway separation loss"""
        return self._rwr_sep_loss_per_hr
    
    @property 
    def rwy_idle_loss(self):
        """Return the current accumulated runway idle loss, 
        which is the time runway was not used when it could have been"""
        return self._rwy_idle_loss_per_hr
    
    @property 
    def runway_throughput(self):
        """Return the array of current accumulated runway throughput"""
        return self._rwy_truput_per_hr
    
    @property 
    def runway_throughput_total(self):
        """Return the runway throughput"""
        return self._rwy_truput_per_hr[-1]
    
    @property 
    def runway_interdep_times(self):
        """Return the runway interdeparture times"""
        return self.rwy_interdep_times
    
   
    def taxi_deice_to_rwy(self,index):
        """Adds minutes to taxi from deice to rwy holding"""
        if self.printout:
            self.event_logging(index, "starts taxi from deice")
        self.total_in_taxi_from_deice  +=1
        #print(f"Total taxi from deice is now:{self.total_in_taxi_from_deice}")
        yield self.env.timeout(self._taxi_time_from_deice[index])
        if self.printout:
            self.event_logging(index, "reached runway holding after deice")
        self.total_in_taxi_from_deice  -=1
        
    def complete_process(self, index):
        """Complete two stage process representing a flight going through the system."""
        #Wait for AOBT: Actual off Block Time:
        yield self.env.timeout(self._AOBTs[index])
        
        #Wait the taxi time to reach deice or runway
        yield self.env.timeout(self._EXOTs[index])
        
        self.update_stats()

        if self.printout:
            self.event_logging(index, "arrives to enter the system")
          
        """Determine if the flight will go via 1st stage (deice) or not:"""
        if self._deice_service_need[index] == 0:
            # Flight bypasses the queue and goes straight to departure
            self._ACZT[index] = self.env.now
            self._AEZT[index] = self.env.now
            self._time_in_deice_system.append((self.env.now, self._total_in_deice_system))
            self._time_in_deice_queue.append((self.env.now, self.total_in_deice_queue))

            if self.printout:
                self.event_logging(index, "bypass deice (no service needed)")
               
            """Proceeds directly to 2nd stage (runway):"""
            yield self.env.process(self.runway_process(index))

        else:
            """Proceeds to 1st stage (deice):"""
            yield self.env.process(self.deice_process(index))
            """Proceeds to runway:"""
            self._last_in_line_taxi_to_rwy = index #flight index ID as last in line to taxi from deice
            self._last_in_line_taxi_to_rwy_time = self.env.now #Time of last taxi from deice
            
            self.trigger_interrupt_event()

            yield self.env.process(self.taxi_deice_to_rwy(index))
            yield self.env.process(self.runway_process(index))

    def deice_process(self,index):
        self._total_in_deice_system += 1
        self.update_stats()
        if self.printout:
                self.event_logging(index, "Enters deicing system")
        
        """Manually requesting and releasing:"""
        request = self.deicing_resource.request()
        yield request
        self.deicing_users[self._flight_ids[index]] = self.env.now  # Track current user with timestamp
      
            # Update the start deice service time when service actually starts
        self._ACZT[index] = self.env.now

        if self.printout:
            self.event_logging(index, "Starts deicing")

        yield self.env.timeout(self._ADITs[index])

        # After deicing:
        if self.printout:
            self.event_logging(index, "Is done deicing - Req Taxi to RWY")
       
        wait_to_taxi = self.wait_before_release_from_deice(index)
       
        results = yield self.env.any_of([self.env.timeout(wait_to_taxi), self.interrupt_event])
    
        self.deicing_resource.release(request)
    
        del self.deicing_users[self._flight_ids[index]]  # Remove user after service
        self._total_in_deice_system -= 1
        self._AEZT[index] = self.env.now
        self.update_stats()
                
    def runway_process(self,index):
        """Entry to the runway system:"""
        self._RTOTs[index] = self.env.now #The time index flight arrived at rwy holding.
        self._total_in_runway_system += 1
        
        if self._number_of_takeoffs==0:
            self._last_takeoff_event_time = self.env.now #For rwy metrics to start now.
            self._first_takeoff_event_time = self.env.now #Stores the time of the 1st takeoff.
        self.update_stats()
        
        if self.printout:
            self.event_logging(index, "Enters the runway system")
            
        with self.runway_resource.request() as runway_request:
            yield runway_request
            # Calculate the runway sep time based on the separation for acft class and SID direction:
            runway_sep_time = self.separation_time(index)

            elapsed_time_last_TO = self.env.now - self._last_takeoff_event_time #time since last T/O

            rwy_idle_time = max(0,elapsed_time_last_TO - runway_sep_time) #Time rwy could have been used.
            self.rwy_idle_time_loss += rwy_idle_time
            remaining_sep_time = max(0,runway_sep_time - elapsed_time_last_TO)
            if runway_sep_time<=1:
                self._rwy_sep_time_loss += 0 #No sep loss as 1 min is optim.
            else: #if separation was 2 minutes from previous
                self._rwy_sep_time_loss += min(1,remaining_sep_time) #counts penalty.
            
            if self.printout:
                self.event_logging(index, f"Rem sep time is {remaining_sep_time} min ")
                
            yield self.env.timeout(remaining_sep_time) #Cleared for takeoff after sep time
            #Interdep time:
            self.rwy_interdep_times[self._number_of_takeoffs] = self.env.now - self._last_takeoff_event_time
            #Updates:
            self._last_takeoff_event_time = self.env.now
            if self.printout:
                self.event_logging(index, f"Cat:{self._categories[index]}, Dir: {self._SIDs[index]} takes off")
                
            # Update the departure time after runway process is completed
            self._ATOTs[index] = self.env.now
            self._number_of_takeoffs += 1
        
            self._rwy_truput_per_hr[self._number_of_takeoffs-1] = self._number_of_takeoffs*60/(self.env.now)
            self._rwy_idle_loss_per_hr[self._number_of_takeoffs-1] = self.rwy_idle_time_loss*60/(self.env.now)
            self._rwr_sep_loss_per_hr[self._number_of_takeoffs-1] =self._rwy_sep_time_loss*60/(self.env.now)
            
            self._total_in_runway_system -= 1
            self.update_stats()

            # Update the last acft cat and SID:
            self._last_acft_cat = self._categories[index]
            self._last_SID = self._SIDs[index]

    def separation_time(self, index):
        #Separation between flights
        if self._last_acft_cat is None or self._last_SID is None:
            return 0  # No previous flight exists

        current_category = self._categories[index]
        current_SID = self._SIDs[index]

        sep_time = 1  # Minimum separation time is 1 minute

        if self._last_acft_cat == 'H' and current_category in ['M', 'L']:
            if self.printout:
                self.event_logging(index, "Needs two min sep after Heavy")
            sep_time = 2

        if self._last_SID == current_SID:
            if self.printout:
                self.event_logging(index, "Needs two min sep due to same SID as prev")
            sep_time = 2

        return sep_time

    def wait_before_release_from_deice(self, index):
        """Wait time before release from deice for SID resequencing. 
        Method to set priority for release from deice to rwy. 
        Default is FCFS or RESEQ_ICE logic below:
        """
        policy = self.policies["Policy_ICE"]
        policy_type = policy["type"]
        max_wait = policy["max_wait_after_deice"]

        #Policy_ICE, max_wait = self.Policy_release_from_deice
        
        if policy_type == "RESEQ_ICE" and self._last_in_line_taxi_to_rwy is not None:
            SID_last_in_line = self._SIDs[self._last_in_line_taxi_to_rwy]
            SID_current = self._SIDs[index]

            if SID_current == SID_last_in_line:
                self._acft_requesting_taxi_from_deice.append(index)
                if self.printout:
                    self.event_logging(index, "waiting for SID sequencing")
                wait_time = max_wait  # Waiting maximum minutes, unless otherwise interrupted.
            else:
                wait_time = 0  # FCFS
        else:
            wait_time = 0  # FCFS

        return wait_time

    def trigger_interrupt_event(self):
        """Function to trigger an interrupt event."""
        self.interrupt_event.succeed()
        # Reset the event:
        self.interrupt_event = self.env.event()


    def update_stats(self):
        
        """Update the time-integrated statistics."""
        current_time = self.env.now
        time_passed = current_time - self._last_event_time
        #Deice stats:
        self._area_in_deice_system += self._total_in_deice_system * time_passed
        self._area_in_deice_queue += self.total_in_deice_queue * time_passed
        self._last_event_time = current_time
        self._time_in_deice_system.append((current_time, self._total_in_deice_system))
        self._time_in_deice_queue.append((current_time, self.total_in_deice_queue))
       
        #Runway Stats:
        self._area_in_runway_system += self._total_in_runway_system * time_passed
        self._area_in_runway_queue += self.total_in_runway_queue * time_passed
        self._last_event_time = current_time
        self._time_in_runway_system.append((current_time, self._total_in_runway_system))
        self._time_in_runway_queue.append((current_time, self.total_in_runway_queue))

   
    @property
    @requires_run
    def ERZTs(self):
        return self._ERZTs

    @property
    @requires_run
    def ADITs(self):
        return self._ADITs
    
    @property
    def TSATs(self):
        return self._TSATs
    
    @TSATs.setter
    def TSATs(self, new_tsats):
        self._TSATs = new_tsats

    @property
    def ATOTs(self):
        return self._ATOTs
    
    @property
    def makespan(self):
        #time from 0 until last takeoff:
        return max(self._ATOTs)
    
    @property
    def deice_service_need(self):
        return self._deice_service_need

    @property
    @requires_run
    def AEZTs(self):
        return self._AEZT

    @property
    @requires_run
    def deice_departure_order(self):
        return self._deice_deice_departure_order

    @property
    @requires_run
    def wait_times_for_deice_queue(self):
        self.wait_times_for_deice = self._ACZT - self._ERZTs
        return self.wait_times_for_deice
    
    @property
    @requires_run
    def wait_times_for_runway_queue(self):
        self.wait_times_for_runway = self._ATOTs - self._RTOTs
        return self.wait_times_for_runway
    

    @property
    @requires_run
    def wait_cost_deice_queue(self):
        self.wait_cost_for_deice = self._wait_b2*self.wait_times_for_deice_queue**2 + self._wait_b1*self.wait_times_for_deice_queue
        #self.wait_cost_for_deice = self.wait_times_for_deice_queue*self._wait_cost_per_mins_b2_b1
        
        return self.wait_cost_for_deice
    
    @property
    @requires_run
    def wait_cost_rwy_queue(self):
        """Assumes same wait cost when waiting at runway as for wait for deice"""
        self.wait_cost_for_rwy = self._wait_b2*self.wait_times_for_runway_queue**2 + self._wait_b1*self.wait_times_for_runway_queue
        return self.wait_cost_for_rwy
    
    @property
    @requires_run
    def wait_cost(self):
        """Calculates the During Taxiing airline delay costs for all flights ref Appendix C.2
        Returns vector with the flights taxiing delay costs
        Note: The quadratic term accounts for sequential delay time already taken at gate
        ref Appendix C.3"""

        wait_times = self.wait_times_for_deice_queue + self.wait_times_for_runway_queue
        self.wait_cost_deice_and_rwy = self._wait_b2*((self.dep_delay_times + wait_times)**2 - self.dep_delay_times**2)  + self._wait_b1*wait_times

        return self.wait_cost_deice_and_rwy


    @property
    @requires_run
    def total_wait_cost_queue(self):
        """The sum of the total wait cost for all flights"""
        self.tot_wait_cost_for_deice = np.sum(self.wait_cost_deice_queue)
        self.tot_wait_cost_for_rwy = np.sum(self.wait_cost_rwy_queue)

        total_wait_cost = np.sum(self.wait_cost)
        return total_wait_cost
        
    
    @property
    @requires_run
    def dep_delay_times(self):
        self.dep_delay = self._AOBTs - self._SOBTs
        return self.dep_delay
    
    @property
    @requires_run
    def delay_cost(self):
        """Calculates the at gate airline delay costs for all flights ref Appendix C.2
        Returns vector with the flights at gate delay costs"""
    
        self.dly_cost = self._delay_b2*self.dep_delay_times**2 + self._delay_b1*self.dep_delay_times
        
        return self.dly_cost
    
    @property
    @requires_run
    def total_delay_cost(self):
        """Totals the at gate delay costs over all flights ref Appendix C.2."""
        self.tot_dly_cost = np.sum(self.delay_cost)
        return self.tot_dly_cost
    
    @property
    @requires_run
    def total_throughput_cost(self):
        """Computes the total cost for the use of runway in accordance with 
        Appendix C.1 Cost Assumptions"""
        runway_cap = 60 #Max theoretical, one dep/min.
        return self.makespan*self.throughput_cost
        
    @property
    @requires_run
    def total_cost(self):
        return self.total_throughput_cost + self.total_delay_cost + self.total_wait_cost_queue
    


    
    @property
    @requires_run
    def total_times_system(self):
        return self._AEZT - self._ERZTs

    @property
    @requires_run
    def mean_L(self):
        return self._area_in_deice_system / self.env.now

    @property
    @requires_run
    def mean_LQ(self):
        return self._area_in_deice_queue / self.env.now
    
    """Metrics:"""
    @property
    @requires_run
    def current_deicing_users(self):
        """Return the current users of the deicing resource."""
        return self.deicing_users

    @requires_run
    def plot_flight_times(self,plot_name,montecarlo=False):
        """Plot of deice arrival, deice service start, and deice dep times
        plot of deice queue and runway queue."""
        
        combined_labels = [f"{fid} ({sid}) ({cat})" for fid, sid, cat in zip(self._flight_ids, self._SIDs, self._categories)]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        y_positions = np.arange(self.num_flights)
        for i in range(self.num_flights):
            if self._deice_service_need[i] == 0:
                ax1.scatter([self._ERZTs[i], self._AEZT[i]], [y_positions[i], y_positions[i]],
                            marker='D', s=100)
            else:
                
                ax1.plot([self._ERZTs[i], self._ACZT[i], self._AEZT[i]], 
                         [y_positions[i], y_positions[i], y_positions[i]])
                ax1.scatter(self._ERZTs[i], y_positions[i], marker='o', s=100)
                ax1.scatter(self._ACZT[i], y_positions[i], marker='x', s=100)
                ax1.scatter(self._AEZT[i], y_positions[i], marker='o', s=100)

        ax1.set_yticks(y_positions)
        ax1.set_yticklabels(combined_labels)
        ax1.set_title(f'{plot_name} Deice arrival (ERZT), Deice Start (ACZT), and Deice Departure Times (AEZT)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Flights (SID) (CAT)')
        ax1.grid(True)
        ax1.minorticks_on()
        ax1.grid(which='minor', axis='x', linestyle=':', linewidth='0.5', color='gray')

        times_system, counts_system = zip(*self._time_in_deice_system)
        times_queue, counts_queue = zip(*self._time_in_deice_queue)
        times_rwy_queue, counts_rwy_queue = zip(*self._time_in_runway_queue)
        
        ax2.step(times_system, counts_system, where='post', label='Number in Deice System', alpha=0.1 if montecarlo else 1.0)
        ax2.step(times_queue, counts_queue, where='post', linestyle='--', label='Number in Deice Queue', alpha=0.1 if montecarlo else 1.0)
        ax2.step(times_rwy_queue, counts_rwy_queue, where='post', linestyle='--', label='Number in Rwy Queue', alpha=0.1 if montecarlo else 1.0)

        ax2.set_title(f'{plot_name} Number in Deice System and Deice Queue and Rwy Queue Over Time')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of Flights')
        ax2.legend()
        ax2.grid(True)
        ax2.minorticks_on()
        ax2.grid(which='minor', axis='x', linestyle=':', linewidth='0.5', color='gray')

        # Align the x-axis min and max:
        min_time = min(times_system)
        max_time = max(times_system)+2 #for some buffer
        ax1.set_xlim(min_time, max_time)
        ax2.set_xlim(min_time, max_time)
        # Save and display plot if not in Monte Carlo mode
        if not montecarlo:
            filename_plot = f'{plot_name}_plot_of_deice_times.png'
            plt.savefig(filename_plot)
            plt.show()
        else:
            print("Line 657 in simulation.py")
        
    @requires_run
    def plot_takeoffs_over_time(self,plot_name):
        # Sort the ATOTs to count takeoffs over time
        sorted_atots = np.sort(self._ATOTs)
        
        # Create an array for the cumulative count of takeoffs
        N_t_ATOT = np.arange(1, len(sorted_atots) + 1)

        # Plot the cumulative takeoffs versus time
        plt.plot(sorted_atots,self.rwy_interdep_times, drawstyle='steps-pre', linestyle='--',label="interdep times")
        plt.plot(sorted_atots, N_t_ATOT, drawstyle='steps-post',linestyle='--', label = "N(t) departures")
        plt.xlabel('Time')
        plt.ylabel('Number of Takeoffs and Interdep times')
        plt.title(f'{plot_name} Takeoffs vs time. Last takeoff time: {sorted_atots[-1]:.4}')
        plt.legend()
        plt.grid(True)
        plt.minorticks_on()
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

        filename_plot = f'{plot_name}_plot_of_takeoffs.png'
        plt.savefig(filename_plot)
        plt.show()



class MonteCarloSimulation:
    def __init__(self, M, flight_schedule, deicing_servers, throughput_cost, policies, printout,mode,max_sim_time, crn_seed = 123,antithetic = False, new_TSATs=None):
        self.M = M
        if mode=="Det":
            #Turns off antithetic for deterministic runs.
            antithetic=False
        
        self.flight_schedule = flight_schedule
        self.mode = mode #Det or Sto
        self._antithetic = antithetic #True or False if antithetic is used.
        self._crn_seed = crn_seed
        
        # Initialize TSATs
        self._original_TSATs = np.array([flight['TSAT'] for flight in self.flight_schedule])

        # Set TSATs to either original or optimized values
        self._TSATs = self._original_TSATs if new_TSATs is None else new_TSATs

        # Initialize airport environment
        self.deicing_servers = deicing_servers
        self.throughput_cost = throughput_cost

        # Initialize other policies
        self.policies = policies
        self.printout = printout
        self.max_sim_time = max_sim_time

    def run(self, M_runs = None, seedchange=None):
        if seedchange:
            self._crn_seed = seedchange
            
        if M_runs: # Adaptive runs
            self.M = M_runs
       
        #Reset CRN seed for the complete set of MC runs:
        np.random.seed(self._crn_seed)
        random.seed(self._crn_seed)

        
        M = self.M

        # Initialize performance metrics
        self.rwy_throughput_array = np.zeros(M)
        self.makespan_array = np.zeros(M)
        self.rwy_sep_loss_array = np.zeros(M)
        self.rwy_idle_loss_array = np.zeros(M)
        self.total_wait_cost_array = np.zeros(M)
        self.total_delay_cost_array = np.zeros(M)
        self.total_cost_array = np.zeros(M)

        if self._antithetic:
            # Inititalize and run:
            self.makespan_x = np.zeros(M)
            self.makespan_x_anti = np.zeros(M)
            # Run:
            self.run_anti()

        
        else: # Crude Monte Carlo:
           
            for runs in range(self.M):
                # Generate an instance of FlightSimulation for each run
                simulation = FlightSimulation(self.flight_schedule, self.deicing_servers, self.throughput_cost, self.policies, self.printout,self.mode,self.max_sim_time,self._antithetic)
                
                # Update the simulation TSATs
                simulation.TSATs = self._TSATs

                # Run the simulation
                simulation.run()

                #Collect performance metrics
                self.makespan_array[runs] = simulation.makespan
                self.rwy_throughput_array[runs] = simulation.runway_throughput_total
                self.rwy_sep_loss_array[runs] = simulation.sep_loss[-1]
                self.rwy_idle_loss_array[runs] = simulation.rwy_idle_loss[-1]
                self.total_wait_cost_array[runs] = simulation.total_wait_cost_queue
                self.total_delay_cost_array[runs] = simulation.total_delay_cost
                self.total_cost_array[runs] = simulation.total_cost

            
    def run_anti(self):
        #Antithetic Monte Carlo
        
        for runs in range(self.M):
            # Stochastic flights generated for each run:
            self.flights, self.flights_anti = generate_flights_anti(self.flight_schedule,self.mode,self._antithetic)
            
            # Generate a separate instance of FlightSimulation for each (anti) run
            simulation_x = FlightSimulation(self.flights, self.deicing_servers, self.throughput_cost, self.policies, self.printout,self.mode,self.max_sim_time,self._antithetic)
            simulation_x_anti = FlightSimulation(self.flights_anti, self.deicing_servers, self.throughput_cost, self.policies, self.printout,self.mode,self.max_sim_time,self._antithetic)
            
            # Update the simulation TSATs
            simulation_x.TSATs = self._TSATs
            simulation_x_anti.TSATs = self._TSATs

            # Run the two antithetic simulations:
            simulation_x.run()
            simulation_x_anti.run()

            #Collect performance metrics
            makespan_x = simulation_x.makespan
            makespan_x_anti = simulation_x_anti.makespan

            self.makespan_x[runs] = simulation_x.makespan
            self.makespan_x_anti[runs] = simulation_x_anti.makespan

            self.makespan_array[runs] = (simulation_x.makespan + simulation_x_anti.makespan)/2
            self.total_cost_array[runs] = (simulation_x.total_cost + simulation_x_anti.total_cost)/2


        if self.printout:
            # Correlation coefficient
            correlation = np.corrcoef(self.makespan_x, self.makespan_x_anti)[0, 1]
            print(f'Correlation between the makespan arrays: {correlation:.4f}')
            print(f'Variance of makespan_x: {np.var(self.makespan_x)}')
            print(f'Variance of makespan_x_anti: {np.var(self.makespan_x_anti)}')
            print(f'Variance of combined makespan: {np.var(self.makespan_array)}')


    def compare_policies(self, tsat_policy_1=None, tsat_policy_2=None, seed_1=None,seed_2=None, objective = "makespan"):
        """Compares two policies in a common difference plot. Sets seed for CRN"""
        # Run simulation with first TSAT policy
        self._TSATs = self._original_TSATs if tsat_policy_1 is None else tsat_policy_1
        self.run(seedchange=seed_1)

        if objective == "makespan":
            objective_policy_1 = self.makespan_array
        else:   #Totalcost:
            objective_policy_1 = self.total_cost_array
        
        # Run simulation with second TSAT policy
        self._TSATs = self._original_TSATs if tsat_policy_2 is None else tsat_policy_2

        self.run(seedchange=seed_2)
        
        if objective == "makespan":
            objective_policy_2 = self.makespan_array
        else:   #Totalcost:
            objective_policy_2 = self.total_cost_array

        # Calculate and return differences for analysis
        objective_diff = objective_policy_1 - objective_policy_2
        self._objective_diff = objective_diff
        print(np.mean(objective_diff))
        return objective_diff
    
    def objective_diff_plots(self,bins = 40, alpha=0.7,color='blue'):
        """Histograms and QQ plots."""
        objective_diff = self._objective_diff
        # Histogram
        plt.hist(objective_diff, bins=bins, alpha=alpha, color=color)
        plt.title('Histogram of objective_diff')
        plt.xlabel('Difference in Objective Value')
        plt.ylabel('Frequency')
        plt.show()

        # Q-Q plot
        stats.probplot(objective_diff, dist="norm", plot=plt)
        plt.title('Q-Q Plot of objective_diff')
        plt.show()


    def minibatch_diff_plots(self, M_batch,N):

        objective_diff = self._objective_diff
        sample_means = np.random.choice(objective_diff , size=M_batch, replace=False)
        # Plot histogram of sample means
        plt.hist(sample_means, bins=40, alpha=0.7, color='blue')
        plt.title(f'Distribution of Sample Means ({M_batch} samples per batch)')
        plt.xlabel('Sample Mean of Objective val diff')
        plt.ylabel('Frequency')
        plt.show()

        # Q-Q plot
        stats.probplot(sample_means, dist="norm", plot=plt)
        plt.title('Q-Q Plot of sample_means diff')
        plt.show()


    def plot_running_average(self, burn_in=10, objective="makespan", plotname="No_name", ax=None, label="No_label"):
        # Calculate the cumulative mean for total cost and makespan
        mu_mc_totalcost_array = np.cumsum(self.total_cost_array) / np.arange(1, len(self.total_cost_array) + 1)
        mu_mc_makespan_array = np.cumsum(self.makespan_array) / np.arange(1, len(self.makespan_array) + 1)
        
        # Calculate standard deviation for total cost and makespan estimator
        std_mu_mc_totalcost_array = np.std(self.total_cost_array) / np.sqrt(np.arange(1, len(self.total_cost_array) + 1))
        std_mu_mc_makespan_array = np.std(self.makespan_array) / np.sqrt(np.arange(1, len(self.makespan_array) + 1))

        # Set up of figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Adjust the x-axis to start from the burn_in index
        x_values = np.arange(burn_in, len(mu_mc_makespan_array))

        # Plot based on the selected objective
        if objective == "makespan":
            ax.plot(x_values, mu_mc_makespan_array[burn_in:], label=f'{label} seed:{self._crn_seed} $\\hat{{\\mu}}_{{MC}}$: {mu_mc_makespan_array[-1]:.2f} SE$_{{\\hat{{\\mu}}_{{MC}}}}$: {std_mu_mc_makespan_array[-1]:.2f}')
            # Plot confidence intervals (95%)
            ax.fill_between(x_values,
                            mu_mc_makespan_array[burn_in:] - 1.96 * std_mu_mc_makespan_array[burn_in:],
                            mu_mc_makespan_array[burn_in:] + 1.96 * std_mu_mc_makespan_array[burn_in:],
                            color='gray', alpha=0.3, label="95% Confidence Interval")
        else:
            ax.plot(x_values, mu_mc_totalcost_array[burn_in:], label=f'{label} $\\hat{{\\mu}}_{{MC}}$: {mu_mc_totalcost_array[-1]:.2f} SE$_{{\\hat{{\\mu}}_{{MC}}}}$: {std_mu_mc_totalcost_array[-1]:.2f}')
            # Plot confidence intervals (95%)
            ax.fill_between(x_values,
                            mu_mc_totalcost_array[burn_in:] - 1.96 * std_mu_mc_totalcost_array[burn_in:],
                            mu_mc_totalcost_array[burn_in:] + 1.96 * std_mu_mc_totalcost_array[burn_in:],
                            color='gray', alpha=0.3, label="95% Confidence Interval")

        # Titles, labels, and grid
        ax.set_title(f'{plotname}')
        ax.set_xlabel('Monte Carlo Simulation Runs')
        ax.set_ylabel('Value')
        ax.grid(True)
        ax.legend()

        # Return the figure and axis
        return fig, ax


    def plot_objective_diff(self, burn_in=10, plotname="Makespan Difference", ax=None, label="Policy Difference"):
        # Calculate the cumulative mean for objective_diff
        objective_diff = self._objective_diff

        mu_mc_objective_diff = np.cumsum(objective_diff) / np.arange(1, len(objective_diff) + 1)
        
        # Calculate standard deviation for the objective_diff estimator
        std_objective_diff = np.std(objective_diff) / np.sqrt(np.arange(1, len(objective_diff) + 1))

        # Set up the figure and axis
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()

        # Adjust the x-axis to start from the burn_in index
        x_values = np.arange(burn_in, len(mu_mc_objective_diff))

        # Plot the cumulative mean of objective_diff
        ax.plot(x_values, mu_mc_objective_diff[burn_in:], label=f'{label} $\\hat{{\\mu}}_{{MC}}$: {mu_mc_objective_diff[-1]:.2f} SE$_{{\\hat{{\\mu}}_{{MC}}}}$: {std_objective_diff[-1]:.2f}')
        
        # Plot 95% confidence intervals
        ax.fill_between(x_values,
                        mu_mc_objective_diff[burn_in:] - 1.96 * std_objective_diff[burn_in:],
                        mu_mc_objective_diff[burn_in:] + 1.96 * std_objective_diff[burn_in:],
                        color='gray', alpha=0.3, label="95% Confidence Interval")

        # Add titles, labels, and grid
        ax.set_title(f'{plotname}')
        ax.set_xlabel('Monte Carlo Simulation Runs')
        ax.set_ylabel('Difference in Objective Value')
        ax.grid(True)
        ax.legend()

        return fig, ax
    
    @property
    def mean_rwy_throughput(self):
        return np.mean(self.rwy_throughput_array)
    
    @property
    def throughput_arr(self):
        return self.rwy_throughput_array
    
    @property
    def mean_makespan(self):
        return np.mean(self.makespan_array)
    
    @property
    def makespan_arr(self):
        return self.makespan_array

    @property
    def mean_total_wait_cost(self):
        return np.mean(self.total_wait_cost_array)
    
    @property
    def mean_total_delay_cost(self):
        return np.mean(self.total_delay_cost_array)
    
    @property
    def mean_total_cost(self):
        return np.mean(self.total_cost_array)
    
    @property
    def totalcost_arr(self):
        return self.total_cost_array
    
    @property
    def mean_rwy_sep_loss(self):
        return np.mean(self.rwy_sep_loss_array)

    @property
    def mean_rwy_idle_loss(self):
        return np.mean(self.rwy_idle_loss_array)
    
    @property
    def TSATs(self):
        return self._TSATs
    
    @TSATs.setter
    def TSATs(self, new_tsats):
        """The setter method in which the optimizer can reset TSATs""" 
        self._TSATs = new_tsats


    
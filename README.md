# Project Code Overview

This repository contains the complete source code for the project, which consists of approximately 4,500 lines of Python code. The project utilizes object-oriented programming and is organized into several modules. Below is an overview of the key components and their purposes.

## Repository Link
The full source code is available on GitHub:  
[h[ttps://github.com/StigP](https://github.com/StigP](https://github.com/StigP/master_thesis))

---

## Project Structure

The code is divided into multiple Python modules, each with a specific role in the project. Here are the main files and their descriptions:

### `inst03n20k5css5b.py`
- Example instance script used to run simulations.
- Configurable to print event logs and generate plots.
- Imports methods from other modules, with dependencies illustrated in a code diagram provided in the project documentation.

### `utils.py`
- Contains utility functions used throughout the project.
- Handles the generation of flight schedules and simulation of flights using antithetic variates for stochastic or deterministic scenarios.

**Approximate code size:** 950 lines

### `simulation.py`
- Implements the `FlightSimulation` class for Discrete Event Simulation.
- Includes the `MonteCarloSimulation` class for running multiple simulations.

**Approximate code size:** 1,000 lines

### `sorting.py`
- Contains a custom sorting algorithm used by the Tailored Optimizer (TS).

**Approximate code size:** 140 lines

### `optimization.py`
- Contains the optimizers used in the project.

**Approximate code size:** 2,100 lines

### `instances_generate.py`
- A generic script for creating and running simulation instances.

**Approximate code size:** 500 lines

### `cost_estimates.py`
- Provides the code for cost estimates as detailed in the project's documentation.

**Approximate code size:** 70 lines

---

## Additional Files

### `requirements.txt`
- Lists all the Python libraries required to run the project.
- Use the following command to install dependencies:
  ```bash
  pip install -r requirements.txt

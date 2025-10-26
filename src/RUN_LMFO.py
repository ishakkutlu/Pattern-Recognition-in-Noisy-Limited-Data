# -*- coding: utf-8 -*-

# LMFO Orchestrator (Layered Multiple Frequency Optimization)
# File: RUN_LMFO.py
# Author: Ishak Kutlu
# Created: 2024-04-28

# Executive summary
# -------------------------------------
# Orchestrates the end-to-end LMFO pipeline across three layers (L1 -> L2 -> L3).
# • Loads/updates training input header (text_dimension, L1 capacity), opens log/output files.
# • Applies L1-only preprocessing choices (scaling mode; optional feed_list pre-bias), then
#   invokes each layer with its own ordering (global vs. relationship-aware), augmentation,
#   and adaptive search settings.
# • Collects multiple unique solutions (L1: from seeds or dummy init; L2/L3: from previous layer outputs),
#   with early-exit controls for zero-profit and duplicates; writes final binary patterns and profits to Solution.txt.
# • Reverse index mapping restores original feature order; decode_solution adds 1-based feature IDs for human-readable logs.
#
# Inputs/Outputs
# -------------------------------------
# In :
# • Training data paths: folderName/fileName, my_input_solution_space
# • Global feature size: text_dimension
#
# • Per-layer windows:                      Phase-1 dimension reductions         Phase-2 dimension reductions
#                                           my_dimension_solution_space (L1)     level_of_dim_reduction_l1
#                                           my_dimension_layer2 (L2)             level_of_dim_reduction_l2
#                                           my_dimension_layer3 (L3)             level_of_dim_reduction_l3
#
# • Per-layer capacities (pattern sizes): 
#                                           my_capacity_solution_space (L1)
#                                           my_capacity_layer2 (L2) 
#                                           my_capacity_layer3 (L3)
#
# • Schedulers:                             p_s (“p” or “s”), s_changer_* (fast/slow),
#                                           k_switch (k oscillation),
#                                           dimension_changer controls per layer (enable + bounds)
#
# • Pre-bias & seeds: feed_list_control, feed_list; fixed_solution_control, fixed_solution_major_list
# • Ordering/repair guards: relationship_control_l1/l2/l3, random_control_l1/l2/l3
#
# Out:
# • Per-layer tuples: (final_solution_output, final_profit, final_cost, profit_count)
# • Pass-throughs for chaining: final_solution, profits_original, weights_original, text_dimension, zero_one_matrix
# • Side effects: Solution.txt (final binary mask + profit lines), Log.txt (run log/metrics)
#
# Key flags
# -------------------------------------
# • avg_sqrt_std_control (L1 only)      : True -> mean/sqrt(std);                                           False -> mean only (scaling baked into L1 outputs).
# • feed_list_control     (L1 only)     : True -> apply cross-run pre-bias using feed_list signal blocks;   False -> skip.
# • relationship_control_l1/l2/l3       : True -> conditional co-activation ordering;                       False -> global ordering.
# • random_control_l1/l2/l3             : True -> protect pinned indices during random edits/repair;        False -> freer edits.
# • fixed_solution_control (L1 only)    : True -> iterate over fixed seeds as initial solutions in L1.      False -> start with dummy placeholder.
#
# Orchestration & data flow
# -------------------------------------
# 1) Setup
#    - modify_input(): rewrites the first line of the training file header with (text_dimension, L1 capacity).
#    - Opens “Solution.txt” (output) and “Log.txt” (run log). Overwrites on each run.
#
# 2) Layer 1 (Solution Space)
#    - For each initial seed (or a placeholder if disabled): run L1 with current p/s, k, and window settings.
#    - Collect unique solutions up to variety_1; skip early on repeated zero-profit or duplicate patterns.
#    - L1 reverse index mapping internally (augmented -> L1 view -> original). 
#    - L1 outputs (solution_space, profits_original, weights_original, text_dimension, zero_one_matrix) feed L2.
#
# 3) Layer 2
#    - For each unique L1 solution: run L2 expansion under its p/s, k, and window schedule.
#    - Collect up to variety_2 with early-exit guards. L2 reverse index mapping internally (augmented -> L2 view -> original).
#    - Pass L2 outputs forward to L3.
#
# 4) Layer 3 (final)
#    - For each unique L2 solution: run L3 as the final expansion stage.
#    - Collect up to variety_3 with early-exit guards. L3 reverse index mapping internally (augmented -> L3 view -> original).
#    - Orchestrator also logs 1-based feature IDs for readability.
#    - Final solutions are written to Solution.txt with their profit values.
#
# Traceability (main calls)
# -------------------------------------
# • File I/O & logging: modify_input, print_log_file, print_solution_to_text_file, decode_solution
# • Schedulers: s_changer_fast / s_changer_slow (p or s), dimension_changer (per layer), k schedule (per layer)
# • Layer runs: Layered_Multiple_Frequency_Optimization_Solution_Space.objective_function(...)
#               Layered_Multiple_Frequency_Optimization_Layer2.objective_function(...)
#               Layered_Multiple_Frequency_Optimization_Layer3.objective_function(...)
#
# Benchmark context (real-world; low SNR & limited data)
# -------------------------------------
# Setup     : 80 items; per event 22 active. 1,200 event records; very low SNR (signal-to-noise ratio).
# Challenge : Astronomical search space (~2×10^19 combinations), scarce data (>99.99% unobserved), weak signals.
# Approach  : Indicative patterns (k out of 22 actives), searched under capacity k.
# Note      : capacity = pattern size = k. This k is distinct from the k used in neighbor oscillation.
# Value     : Extracts robust, stable k-patterns under heavy noise; preserves context across layers;
#             yields reproducible, actionable patterns for decision support.
#
# Notes
# -------------------------------------
# • Data preparation: raw events are processed and binarized (0/1 per feature) upstream in the external 'control_file'; LMFO expects pre-binarized inputs.
# • Splits: train/validation/test partitions (and any folds) are defined in 'control_file'; RUN_LMFO and layer files operate on the training split.
# • Evaluation: validation & generalization (val/test) run in the external 'control_file' (outside LMFO) to maintain separation of concerns.


# -------------------------------------------------------------------------
# Import core LMFO layers (Layer 1–3)
# -------------------------------------------------------------------------

from LMFO_Solution_Space import Layered_Multiple_Frequency_Optimization_Solution_Space # Layer 1
from LMFO_L2 import Layered_Multiple_Frequency_Optimization_Layer2 # Layer 2
from LMFO_L3 import Layered_Multiple_Frequency_Optimization_Layer3 # Layer 3
import copy

# -------------------------------------------------------------------------
# Input file definitions (used as training set for pattern generation)
# -------------------------------------------------------------------------

folder_name_solution_space = "input_folder" 
file_name_solution_space = "input_solution_space.txt" 
my_input_solution_space = "input_folder/input_solution_space.txt" 

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------

f_log = open("Log.txt", "w") # log file for run information (overwrites each run)
def print_log_file(log_str):
    f_log.writelines(str(log_str))

# -------------------------------------------------------------------------
# Startup banner (console + log)
# -------------------------------------------------------------------------

print()
print('Layered Multiple Frequency Optimization - LMFO ' + ' [Developed by Ishak Kutlu]')
print()
print_log_file("\n" + 'Layered Multiple Frequency Optimization - LMFO ' + ' [Developed by Ishak Kutlu]' + "\n")

f = open("Solution.txt", "w") # output file where final patterns will be written

# -------------------------------------------------------------------------
# Functions
# -------------------------------------------------------------------------

def modify_input(my_input_solution_space, text_dimension, my_capacity_solution_space):
    # Update the first line of the input file.
    # The header defines:
    # - text_dimension             -> problem dimension (number of features)
    # - my_capacity_solution_space -> pattern size for Layer 1
    # This sets the initial parameters of the solution space for Layer 1.

    # remove existing header (keep training data intact)
    with open(my_input_solution_space, 'r+') as source_file: 
        lines = source_file.readlines()
        source_file.seek(0)
        source_file.truncate()
        source_file.writelines(lines[1:])  
        source_file.close()
    # read remaining training data
    with open(my_input_solution_space, 'r') as original: 
        data = original.read()
        original.close()
    # write new header (<text_dimension> <capacity>) and restore training data
    with open(my_input_solution_space, 'w') as modified:
        modified.write(str(text_dimension)+" "+ str(my_capacity_solution_space)+"\n" + data)
        modified.close()

def convert_numeric_solution(solution):
    # Ensure the solution vector is numeric (0/1).
    # Converts items from boolean to integer.
    # Example: [True, False, True] -> [1, 0, 1]

    numeric_solution = []
    for item in solution:
        numeric_solution.append(int(item))
    return numeric_solution

def print_solution_to_text_file(solution, profit_count, profit, dimension):
    # Write a binary solution vector with its profit value to Solution.txt.
    # Format:
    # 0, 1, 0, 1, ..., 0, profit value

    y = 0
    for x in solution:
        if profit_count > 0:
            if y < dimension - 1:
                f.writelines(str(x)+", ")
            else:
                f.writelines(str(x)+", ")
                f.writelines(str(profit) + "\n")
            y+=1

def decode_solution(final_solution_output):
    # Decode a binary solution vector into 1-based indices of selected items.
    # Note: Indices are 1-based because they represent feature IDs (1–80).
    # Example: [0, 1, 0, 1, 0, 0, 1] -> [2, 4, 7]

    numeric_list = []
    x = 0
    for item in final_solution_output:
        if item == 1:
            numeric_list.append(x+1)
        x += 1
    return numeric_list

def s_changer_fast(i, s, s_modifier, s_up, s_up_limit, s_down_limit):
    # Fast scheduler for search parameters (e.g., s, p, k).
    # Oscillates a parameter between lower and upper limits.
    # Step size is adapted (×2 / ÷2) to accelerate changes and escape plateaus.

    if i % 1 == 0:
        if s_up == True:
            s += s_modifier
            if (2*s) > s_up_limit:
                s_modifier = s_up_limit
            else:
                s_modifier = copy.deepcopy(2*s)
            if s > s_up_limit:
                #s = s_down_limit
                s -= s_modifier
                s_up = False
                if int((s/2)) < s_down_limit:
                    s_modifier = s_down_limit
                else:
                    s_modifier = int(copy.deepcopy(s/2))
        else:
            s -= s_modifier
            if int((s/2)) < s_down_limit:
                s_modifier = s_down_limit
            else:
                s_modifier = int(s/2) 
            if s < s_down_limit:
                s += s_modifier
                s_up = True
                if (2*s) > s_up_limit:
                    s_modifier = s_up_limit
                else:
                    s_modifier = (2*s)
    return s_modifier, s, s_up

def s_changer_slow(i, s, s_modifier, s_up, s_up_limit, s_down_limit):
    # Slow scheduler for search parameters (e.g., s, p, k).
    # Oscillates a parameter between lower and upper limits 
    # with small, fixed step size (slower exploration than fast version).

    if i % 1 == 0:
        if s_up == True:
            s += s_modifier
            if s > s_up_limit:
                s = s_up_limit - s_modifier
                #s = s_down_limit
                s_up = False
        else:
            s -= s_modifier 
            if s < s_down_limit:
                if s_changer_direction_up_limit == False:
                    s = s_down_limit + s_modifier
                else:
                    s = s_up_limit
                s_up = True
    return s, s_up

def dimension_changer(d, d_modifier, d_up, d_up_limit, d_down_limit):
    # Scheduler for dimension parameter 'd'.
    # 'd' controls the active window size in the search space.
    # This function oscillates 'd' between lower and upper limits,
    # switching direction when boundaries are reached.

    if d_up == True:
        d += d_modifier
        if d > d_up_limit:
            d = d_up_limit - d_modifier
            #d = d_down_limit
            d_up = False
    else:
        d -= d_modifier 
        if d < d_down_limit:
            if d_changer_direction_up_limit == False:
                d = d_down_limit + d_modifier
            else:
                d = d_up_limit
            d_up = True        
    return d, d_up

# -------------------------------------------------------------------------
# Logging & Screen Printing Functions
# -------------------------------------------------------------------------

# Print and log solution status for the given layer.
# status:
#     "U_S" = unique solution
#     "s_S" = same solution (duplicate solution detected)
#     "z_s" = zero-profit solution (no gain achieved)
def screen_printer(c, layer_no, my_capacity, profit_count):
    print("U_S - ", str(c) + ") ", str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))
    print_log_file("\n" + "U_S - " + str(c) + ") " + str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))
    
def screen_printer_same_solution(c, layer_no, my_capacity, profit_count):
    print("s_S - ", str(c) + ") ", str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))
    print_log_file("\n" + "s_S - " + str(c) + ") " + str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))

def screen_printer_zero_solution(c, layer_no, my_capacity, profit_count):
    print("z_s - ", str(c) + ") ", str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))
    print_log_file("\n" + "z_s - " + str(c) + ") " + str(layer_no) + " - Set Size: " + str(my_capacity) + " - Profit: " + str(profit_count))


# -------------------------------------------------------------------------
# Dimension window scheduler controls (Layer 1–3)
# -------------------------------------------------------------------------

# Global behavior
d_changer_direction_up_limit = False  # If True: on hitting lower bound, jump toward up_limit (upward bias)

# L1 — dimension window control
dimension_changer_control_l1 = False  # enable/disable dimension oscillation at Layer 1
d1_up = True                          # current direction (True=up, False=down)
d_modifier1 = 5                       # step size
d1_up_limit = 55                      # upper bound for d
d1_down_limit = 50                    # lower bound for d

# L2 — dimension window control
dimension_changer_control_l2 = False  # enable/disable dimension oscillation at Layer 2
d2_up = True
d_modifier2 = 5
d2_up_limit = 55
d2_down_limit = 50

# L3 — dimension window control
dimension_changer_control_l3 = False  # enable/disable dimension oscillation at Layer 3
d3_up = True
d_modifier3 = 5
d3_up_limit = 55
d3_down_limit = 50


# ============================== COPY SECTION — START ==============================
# Please read LMFO_Reproduce_README for copy section.

# -------------------------------------------------------------------------
# Core LMFO configuration: problem dimension and pattern sizes (capacities) per layer
# -------------------------------------------------------------------------

# Global problem dimension (total number of features)
text_dimension = 80

# Layer 1 configuration
my_dimension_solution_space = 50      # active window size at L1 (subset of features)
level_of_dim_reduction_l1 = 20        # reduction factor applied at L1
my_capacity_solution_space = 7        # pattern size at L1 (must be > 0)

# Layer 2 configuration
my_dimension_layer2 = 50
level_of_dim_reduction_l2 = 40
my_capacity_layer2 = 9                # pattern size at L2 (0 disables L2; must be > L1 pattern size)

# Layer 3 configuration
my_dimension_layer3 = 50
level_of_dim_reduction_l3 = 50
my_capacity_layer3 = 10               # pattern size at L3 (0 disables L3; must be > L2 pattern size)

# Update the input file header (problem dimension + L1 pattern size)
modify_input(my_input_solution_space, text_dimension, my_capacity_solution_space)


# -------------------------------------------------------------------------
# Objective & scheduling switches
# -------------------------------------------------------------------------

p_s = "p"  # choose objective variant: "p" or "s"

avg_sqrt_std_control = True
# True  -> use mean / sqrt(std) weighting
# False -> use mean only

s_changer_direction_up_limit = False
# When reversing at the lower bound:
#   True  -> jump to (s = s_up_limit)
#   False -> jump to (s = s_down_limit + s_modifier)

k_switch = True  # enable/disable k oscillation (scheduler)


############## LAYER 1 ##############
# k schedule (if k_switch=True)
k1_up = False
k_modifier1 = 1
k1_up_limit = 6
k1_down_limit = 4
k1_default = 5
k1_up_default = False

# s/p schedule (if p_s = "p" then p schedule, if p_s = "s" then s schedule)
s1_up = True
s_modifier1 = 2
s1_up_limit = 61
s1_down_limit = 57

# early-exit and duplicate-solution thresholds
eleminate_zero_sol_1 = 4        # consecutive zero-profit attempts before skipping
eleminate_same_sol_1 = 10       # duplicate solutions before skipping

# generation controls
travel_control_l1 = True        # if True, move s/p after each unique solution; if False, keep parameters fixed
random_control_l1 = True        # if True, keep seed solution unchanged; if False, allow modifications
relationship_control_l1 = True  # if True, enforce relational constraints; if False, ignore them

variety_1 = 3  # number of unique solutions to extract at L1 per seed

############## LAYER 2 ##############
# k schedule (if k_switch=True)
k2_up = False
k_modifier2 = 1
k2_up_limit = 4
k2_down_limit = 2
k2_default = 3
k2_up_default = False

# s/p schedule
s2_up = True
s_modifier2 = 2
s2_up_limit = 41
s2_down_limit = 37

# early-exit and duplicate-solution thresholds
eleminate_zero_sol_2 = 4
eleminate_same_sol_2 = 10

# generation controls
travel_control_l2 = True
random_control_l2 = True
relationship_control_l2 = True

variety_2 = 4  # unique solutions to extract at L2 per L1 input

############## LAYER 3 ##############
# k schedule (if k_switch=True)
k3_up = False
k_modifier3 = 1
k3_up_limit = 3
k3_down_limit = 1
k3_default = 3
k3_up_default = False

# s/p schedule
s3_up = True
s_modifier3 = 2
s3_up_limit = 11
s3_down_limit = 7

# early-exit and duplicate-solution thresholds
eleminate_zero_sol_3 = 4
eleminate_same_sol_3 = 10

# generation controls
travel_control_l3 = True
random_control_l3 = True
relationship_control_l3 = False

variety_3 = 3  # unique solutions to extract at L3 per L2 input

# ============================== COPY SECTION — END ==============================


# -------------------------------------------------------------------------
# External feed list (run-to-run memory)
# -------------------------------------------------------------------------

# When enabled, applied once at startup (in Layer 1) to reweight the global profit matrix
# derived from training data. 
# The reweighted matrix is propagated to Layers 2 and 3.
# This biases the search toward co-activations around the given signal blocks.
# Works as a memory transfer mechanism -> earlier runs guide later runs.
feed_list_control = False   # True -> use feed_list; False -> ignore
feed_list =             [

[16, 21, 53, 77],
[2, 12, 20, 73],
[4, 65, 73, 77],
[12, 16, 21, 50],
[18, 21, 26, 69],
[16, 33, 51, 59]

]   # Seeds reference feature columns (1–80); reweighting uses their 0/1 vectors over all samples (co-activation), not hard constraints.


# -------------------------------------------------------------------------
# External fixed/initial solutions (mandatory seeds)
# -------------------------------------------------------------------------

# When enabled, Layer 1 iterates over these seeds as initial solutions.
# Preservation during expansion/repair is controlled by 'random_control_l1':
#   True -> keep seed items unchanged
#   False -> allow modifications
fixed_solution_control = True   # True → use fixed seeds; False → ignore 
fixed_solution_major_list = [
    

[16, 21, 53, 77],
[2, 12, 20, 73],
[4, 65, 73, 77]

]   # feature IDs (1–80)


'''
[16, 21, 53, 77]

[4, 50, 69, 77],
[21, 42, 46, 77],
[18, 21, 42, 77]

[21, 26, 42, 46],
[19, 59, 71, 73],
[12, 16, 34, 73]

[16, 34, 73, 77],
[12, 16, 21, 50],
[21, 33, 42, 77]

[12, 16, 73, 77],
[16, 21, 50, 77],
[4, 7, 69, 77]

[18, 33, 42, 77],
[12, 50, 69, 77],
[12, 16, 50, 73]

[21, 34, 73, 77],
[12, 16, 53, 73],
[21, 50, 73, 77],
[7, 26, 73, 77]


Random seeds

[2, 21, 23, 58], [18, 45, 55, 80], [27, 30, 58, 75], [16, 43, 44, 76], [8, 59, 62, 65],
[7, 20, 56, 62], [27, 33, 35, 45], [17, 46, 68, 76], [10, 33, 73, 78], [31, 40, 46, 61],
[40, 44, 61, 72], [1, 4, 12, 59], [47, 55, 60, 66], [28, 39, 72, 77], [3, 30, 42, 52],
[7, 9, 54, 61], [4, 10, 45, 74], [22, 58, 66, 74], [27, 35, 51, 73], [13, 35, 56, 73],
[21, 32, 53, 63], [40, 51, 53, 63], [21, 37, 79, 80], [19, 23, 32, 50], [3, 8, 9, 26],
[43, 45, 48, 51], [1, 35, 62, 78], [49, 53, 59, 80], [24, 62, 70, 71], [2, 6, 45, 63],
[34, 53, 57, 71], [5, 40, 46, 50], [13, 42, 67, 79], [2, 22, 70, 80], [3, 15, 31, 40],
[32, 61, 71, 73], [53, 56, 62, 72], [7, 33, 67, 79], [5, 26, 39, 44], [2, 9, 21, 25],
[3, 31, 34, 46], [9, 34, 47, 49], [34, 53, 59, 71], [18, 37, 46, 74], [17, 42, 55, 76],
[2, 13, 19, 47], [2, 39, 75, 80], [31, 35, 40, 54], [41, 42, 66, 70], [1, 18, 47, 78],
[15, 17, 23, 64], [1, 10, 30, 63], [19, 44, 51, 66], [5, 23, 25, 40], [1, 9, 22, 29],
[15, 49, 51, 78], [26, 29, 53, 77], [21, 44, 52, 62], [17, 33, 60, 79], [15, 24, 28, 48],
[7, 10, 47, 49], [2, 10, 20, 34], [6, 9, 17, 49], [10, 41, 57, 67], [4, 14, 17, 48],
[9, 50, 76, 77], [38, 66, 77, 78], [8, 45, 66, 76], [18, 27, 55, 61], [10, 16, 43, 50],
[19, 29, 38, 47], [11, 41, 69, 77], [35, 36, 49, 65], [19, 21, 36, 52], [9, 33, 48, 77],
[1, 13, 25, 78], [1, 34, 66, 67], [24, 52, 67, 72], [5, 7, 13, 43], [5, 6, 40, 74],
[13, 24, 44, 64], [23, 30, 44, 59], [6, 35, 55, 80], [5, 12, 61, 66], [3, 40, 46, 50],
[17, 35, 48, 58], [22, 38, 57, 68], [33, 35, 43, 65], [13, 27, 31, 74], [13, 15, 38, 67],
[18, 47, 73, 76], [21, 37, 72, 76], [13, 16, 49, 64], [30, 49, 62, 78], [10, 26, 54, 57],
[18, 53, 54, 61], [10, 40, 45, 57], [26, 30, 73, 78], [49, 50, 51, 71], [9, 31, 47, 49],
[25, 47, 49, 60], [3, 20, 28, 35], [56, 62, 71, 77], [7, 10, 62, 80], [3, 8, 19, 65],
[7, 10, 54, 76], [3, 22, 51, 66], [41, 46, 50, 65], [16, 45, 50, 59], [8, 9, 12, 58],
[15, 42, 67, 70], [24, 34, 40, 74], [24, 39, 53, 65], [3, 22, 24, 76], [14, 27, 30, 48],
[14, 27, 46, 66], [22, 29, 49, 64], [2, 12, 25, 32], [2, 36, 47, 48], [1, 32, 51, 54],
[21, 25, 36, 57], [12, 37, 52, 65], [2, 23, 51, 58], [17, 52, 62, 66], [16, 25, 29, 64],
[5, 19, 52, 60], [25, 46, 57, 66], [26, 40, 61, 72], [10, 55, 67, 73], [2, 43, 48, 63],
[2, 21, 42, 79], [25, 44, 50, 78], [16, 29, 69, 80], [29, 39, 42, 73], [7, 29, 45, 46],
[26, 34, 40, 53], [13, 22, 42, 61], [16, 54, 65, 68], [20, 27, 29, 76], [20, 22, 43, 48],
[16, 60, 65, 71], [16, 74, 76, 79], [6, 33, 45, 73], [9, 60, 72, 78], [1, 16, 51, 68],
[14, 22, 42, 80], [1, 43, 49, 72], [15, 39, 55, 58], [42, 46, 59, 68], [5, 23, 45, 74],
[2, 13, 38, 57], [27, 30, 54, 69], [1, 14, 42, 74], [37, 42, 57, 80], [6, 14, 41, 64],
[12, 25, 74, 77], [15, 27, 53, 56], [19, 45, 64, 75], [43, 52, 54, 74], [12, 13, 16, 69],
[19, 40, 49, 76], [8, 56, 64, 78], [17, 33, 35, 65], [21, 31, 44, 62], [2, 38, 65, 80],
[15, 22, 71, 72], [6, 11, 19, 55], [12, 32, 54, 65], [3, 18, 66, 71], [28, 34, 61, 65],
[12, 43, 49, 76], [21, 42, 66, 75], [26, 27, 59, 78], [2, 14, 43, 66], [12, 48, 77, 78],
[9, 49, 55, 76], [22, 45, 60, 70], [8, 26, 51, 57], [49, 58, 61, 72], [16, 30, 69, 75],
[13, 31, 62, 78], [13, 28, 49, 60], [5, 22, 29, 53], [23, 32, 62, 75], [51, 56, 59, 66],
[27, 28, 38, 45], [32, 50, 54, 70], [4, 9, 31, 53], [1, 2, 19, 73], [12, 19, 21, 28],
[38, 44, 61, 64], [27, 37, 70, 74], [8, 29, 47, 70], [33, 41, 45, 80], [4, 56, 61, 73],
[8, 53, 59, 68], [30, 45, 47, 58], [5, 11, 45, 65], [15, 24, 38, 45], [11, 27, 43, 67],
[22, 38, 44, 74], [24, 28, 51, 75], [3, 45, 52, 70], [29, 31, 45, 46], [16, 26, 35, 51],
[8, 16, 69, 78], [8, 46, 54, 70], [5, 44, 69, 72], [12, 33, 44, 50], [3, 28, 35, 70],
[17, 21, 39, 66], [8, 50, 52, 60], [19, 25, 33, 48], [2, 12, 51, 64], [3, 8, 47, 51], [15, 29, 35, 61]


LMFO-Optimized seeds 

[46, 59, 65, 71], [4, 46, 59, 71], [12, 53, 59, 72], [4, 33, 59, 71], [16, 33, 51, 59],
[16, 59, 65, 71], [37, 46, 59, 65], [4, 12, 59, 71], [21, 42, 46, 53], [16, 46, 51, 71],
[4, 50, 59, 71], [4, 12, 53, 59], [4, 18, 46, 59], [16, 46, 65, 71], [16, 37, 46, 65],
[12, 21, 42, 53], [16, 33, 59, 71], [19, 46, 59, 73], [4, 12, 71, 72], [35, 46, 53, 71],
[21, 42, 53, 77], [12, 59, 71, 72], [19, 21, 42, 77], [18, 46, 59, 71], [16, 34, 37, 73],
[19, 34, 37, 73], [7, 53, 59, 71], [18, 21, 33, 77], [33, 51, 59, 71], [18, 42, 46, 77],
[16, 46, 51, 59], [46, 65, 69, 73], [18, 21, 42, 53], [21, 48, 53, 77], [16, 19, 34, 73],
[16, 21, 50, 53], [21, 42, 73, 77], [50, 53, 69, 77], [50, 53, 73, 77], [19, 21, 53, 77],
[18, 46, 65, 77], [46, 53, 59, 71], [7, 12, 53, 69], [18, 26, 42, 46], [16, 19, 37, 73],
[16, 19, 46, 59], [4, 7, 53, 77], [16, 19, 59, 73], [36, 46, 65, 69], [36, 46, 69, 73],
[42, 46, 65, 77], [7, 33, 46, 73], [4, 50, 53, 77], [16, 19, 53, 73], [12, 50, 53, 73],
[35, 53, 59, 71], [12, 50, 73, 77], [12, 50, 53, 69], [33, 46, 65, 77], [12, 21, 50, 53],
[16, 50, 53, 77], [21, 50, 53, 73], [12, 34, 73, 77], [18, 21, 33, 42], [21, 53, 73, 77],
[12, 16, 21, 53], [12, 16, 34, 77], [16, 50, 53, 73], [16, 21, 34, 50], [12, 16, 50, 77],
[21, 34, 53, 77], [12, 16, 50, 53], [34, 53, 73, 77], [16, 21, 34, 77], [21, 34, 50, 77],
[16, 51, 53, 73], [16, 21, 34, 73], [21, 34, 50, 73], [12, 21, 53, 73], [12, 34, 53, 77],
[12, 34, 53, 73], [16, 34, 50, 77], [16, 34, 69, 73], [12, 16, 21, 73], [16, 21, 73, 77],
[16, 34, 53, 73], [7, 21, 69, 73], [16, 21, 34, 53], [16, 34, 50, 53], [12, 34, 50, 73],
[34, 50, 73, 77], [12, 16, 34, 50], [33, 50, 53, 59], [12, 16, 53, 77], [16, 18, 37, 59],
[4, 7, 59, 77], [7, 72, 73, 77], [21, 26, 42, 71], [2, 50, 71, 77], [18, 37, 50, 59],
[16, 42, 44, 77], [33, 53, 71, 77], [2, 59, 71, 77], [33, 50, 59, 77], [2, 33, 50, 53],
[33, 50, 53, 71], [26, 72, 73, 77], [7, 19, 26, 77], [7, 19, 26, 73], [7, 19, 73, 77],
[19, 72, 73, 77], [19, 26, 72, 77], [19, 26, 72, 73], [12, 33, 71, 77], [53, 59, 69, 77],
[7, 19, 26, 72], [12, 50, 53, 59], [4, 7, 33, 53], [18, 34, 53, 71], [16, 46, 59, 71],
[18, 21, 42, 46], [18, 33, 46, 65], [12, 50, 53, 77], [12, 21, 34, 53], [4, 12, 53, 72],
[4, 53, 59, 72], [7, 33, 53, 71], [33, 46, 59, 71], [37, 46, 59, 71], [37, 46, 65, 71],
[19, 33, 46, 53], [16, 37, 59, 71], [4, 33, 50, 71], [4, 7, 50, 69], [7, 33, 59, 71],
[7, 33, 53, 59], [46, 51, 59, 71], [4, 12, 59, 72], [4, 53, 71, 72], [12, 53, 69, 77],
[16, 37, 46, 71], [33, 53, 59, 71], [12, 34, 42, 53], [12, 53, 71, 72], [16, 51, 59, 71],
[16, 59, 71, 73], [37, 59, 65, 71], [19, 46, 59, 71], [16, 37, 46, 59], [53, 59, 71, 72],
[18, 42, 46, 53], [33, 46, 51, 59], [34, 46, 59, 71], [16, 46, 59, 65], [12, 21, 34, 42],
[16, 37, 59, 65], [21, 34, 42, 53], [12, 53, 73, 77], [18, 21, 26, 46], [4, 59, 71, 72],
[12, 53, 59, 71], [21, 42, 65, 77], [7, 50, 53, 77], [7, 33, 69, 77], [4, 12, 50, 53],
[21, 42, 46, 72], [18, 21, 46, 53], [18, 21, 26, 42], [33, 46, 51, 71], [18, 21, 33, 65],
[19, 42, 53, 77], [21, 50, 53, 77], [21, 26, 42, 46], [19, 59, 71, 73], [12, 16, 34, 73],
[4, 50, 69, 77], [21, 42, 46, 77], [18, 21, 42, 77], [16, 34, 73, 77], [12, 16, 21, 50],
[21, 33, 42, 77], [12, 16, 73, 77], [16, 21, 53, 77], [4, 7, 69, 77], [16, 21, 50, 77],
[18, 33, 42, 77], [12, 50, 69, 77], [12, 16, 50, 73], [21, 34, 73, 77], [12, 16, 53, 73],
[21, 50, 73, 77], [7, 26, 73, 77], [12, 21, 34, 77], [21, 69, 73, 77], [7, 26, 72, 73],
[19, 26, 73, 77], [5, 37, 69, 71], [26, 34, 42, 71], [7, 19, 72, 77], [7, 26, 72, 77],
[7, 19, 72, 73], [18, 35, 42, 59], [4, 18, 46, 71], [33, 50, 59, 71], [16, 37, 65, 71],
[16, 19, 59, 71], [16, 33, 51, 71], [42, 46, 53, 72], [21, 42, 53, 72], [16, 19, 34, 37],
[16, 46, 71, 73], [21, 34, 50, 53], [21, 34, 53, 73], [4, 33, 50, 59]

'''


# -------------------------------------------------------------------------
# Global accumulators & trackers (used across layers)
# -------------------------------------------------------------------------

profit_solution_space = 0
profit_count_solution_space = 0
profit_layer2 = 0
profit_count_layer2 = 0
profit_layer3 = 0
profit_count_layer3 = 0

unique_solution_list1 = []
unique_solution_list2 = []
unique_solution_list3 = []
general_unique_solution_list1 = []
general_unique_solution_list2 = []
general_unique_solution_list3 = []


# -------------------------------------------------------------------------
# Main extraction loops and run layers
# -------------------------------------------------------------------------

##################### LAYER 1 #####################

# Run Layer 1 only if enabled (pattern size > 0)
if my_capacity_solution_space > 0:
    c = 0                           # global step counter (for logging)
    l1 = 0                          # L1-local step counter
    expected_l1 = 0                 # rolling expected unique count
    
    layer_no = "LAYER 1"
    print()
    print(layer_no)
    print_log_file("\n" + layer_no)
    
    # Determine how many initial seeds to iterate (fixed list vs single run)
    if fixed_solution_control == True and len(fixed_solution_major_list) > 0:
        loop_l1 = len(fixed_solution_major_list)
    else:
        loop_l1 = 1
        fixed_solution_major_list = [[0]]   # placeholder seed (no fixed items)
    
    # Iterate over initial seeds (fixed) or a single placeholder seed
    for fixed_solution in fixed_solution_major_list:

        # Reset per-seed trackers
        unique_counter_l1 = 0
        zero_solution_list_1 = []          # for early exit (zero-profit)
        same_solution_list_1 = []          # for early exit (duplicates)
        unique_solution_list1_sS_control = []
        unique_solution_list1_zs_control = []
        s1 = s1_up_limit                   # L1 scheduler start (p/s)
        
        # Reset k to defaults (if k oscillation is enabled)
        k1 = k1_default
        k1_up = k1_up_default
        
        # Log the initial seed (if any)
        if fixed_solution_major_list[0][0] != 0:
            print()
            print("Initial Solution: ", fixed_solution)
            print_log_file("\n" + "\n" + "Initial Solution: " + str(fixed_solution))
        
        unique_solution_list1 = []         # uniques collected for this seed
        
        # Keep extracting until we reach target variety for this seed
        while (len(unique_solution_list1) < variety_1):
            l1 += 1
            c += 1
            
            # Reset duplicate tracker when a new unique appears
            if len(unique_solution_list1) > len(unique_solution_list1_sS_control):
                same_solution_list_1 = []
                unique_solution_list1_sS_control.append(unique_solution_list1[-1])
            
            # Reset zero-profit tracker when a new unique appears
            if len(unique_solution_list1) > len(unique_solution_list1_zs_control):
                zero_solution_list_1 = []
                unique_solution_list1_zs_control.append(unique_solution_list1[-1])
            
            # Fixed seed handling for this iteration (or empty if not enabled)
            if fixed_solution_control == True:
                fixed_solution_list = fixed_solution
            else:
                fixed_solution_list = []
            
            # Early exit for this seed if thresholds are hit (zero/duplicate)
            if len(zero_solution_list_1) == eleminate_zero_sol_1 or len(same_solution_list_1) == eleminate_same_sol_1:
                expected_l1 += variety_1 - unique_counter_l1
                if len(general_unique_solution_list1) <= variety_1 * loop_l1:
                    print("Number of unique solutions in layer 1: Expected = ", expected_l1, " Extracted = ", len(general_unique_solution_list1))
                    print_log_file("\n" + "Number of unique solutions in layer 1: Expected = " + str(expected_l1) + " Extracted = " + str(len(general_unique_solution_list1)))
                break

            # Define pattern size for the next layer.
            # If 0 -> no further layer is run, L1 outputs become the final results.
            next_capacity = my_capacity_layer2
            
            # Run L1 with current controls (feed, random, relationship, scaling)
            problem_solution_space = Layered_Multiple_Frequency_Optimization_Solution_Space(folder_name_solution_space, 
                                                                                    file_name_solution_space, my_dimension_solution_space, fixed_solution_list,
                                                                                    level_of_dim_reduction_l1, feed_list, feed_list_control, 
                                                                                    random_control_l1, relationship_control_l1, avg_sqrt_std_control)
            
            # Optionally oscillate L1 dimension window before running objective
            if dimension_changer_control_l1 == True:
                d1 = my_dimension_solution_space
                d1, d1_up = dimension_changer(d1, d_modifier1, d1_up, d1_up_limit, d1_down_limit)
                my_dimension_solution_space = d1 
                
            # Run objective with selected schedule (p or s) and current k
            if p_s == "p":
                (solution_space, final_solution_output, profit_solution_space, cost_solution_space, profit_count_solution_space, 
                profits_original, weights_original, text_dimension, zero_one_matrix) = problem_solution_space.objective_function(p = s1, k = k1) #update 13.12.2024 k = k1
            else:
                (solution_space, final_solution_output, profit_solution_space, cost_solution_space, profit_count_solution_space, 
                profits_original, weights_original, text_dimension, zero_one_matrix) = problem_solution_space.objective_function(s = s1, k = k1) #update 13.12.2024 k = k1
            
            # Case: zero-profit outcome -> adjust schedulers and continue
            if profit_count_solution_space == 0:
                # First push s/p toward mid-range once, then continue with slow/fast scheduler
                if s1 < int((s1_up_limit + s1_down_limit)/2)-1 or s1 == s1_up_limit:
                    s1 = int((s1_up_limit + s1_down_limit)/2)-1
                    if s1 % 2 == 0:
                        s1 = s1+1
                s1, s1_up = s_changer_slow(l1, s1, s_modifier1, s1_up, s1_up_limit, s1_down_limit)
                zero_solution_list_1.append(solution_space)
                screen_printer_zero_solution(c, layer_no, my_capacity_solution_space, profit_count_solution_space)
                # Optionally oscillate k as well (if enabled)
                if k_switch == True:
                    k1_up = True
                    k1, k1_up = s_changer_slow(l1, k1, k_modifier1, k1_up, k1_up_limit, k1_down_limit)
                continue
            else:
                # After a non-zero result, reset k to defaults (if k oscillation is enabled)
                if k_switch == True:
                    k1 = k1_default
                    k1_up = k1_up_default
                
            # Case: duplicate solution -> adjust schedulers and continue
            if solution_space in general_unique_solution_list1:
                s1, s1_up = s_changer_slow(l1, s1, s_modifier1, s1_up, s1_up_limit, s1_down_limit)
                if k_switch == True:
                    k1, k1_up = s_changer_slow(l1, k1, k_modifier1, k1_up, k1_up_limit, k1_down_limit)                 
                same_solution_list_1.append(solution_space)
                screen_printer_same_solution(c, layer_no, my_capacity_solution_space, profit_count_solution_space)
                continue
            else:
                # Unique solution -> record and move (or pin) scheduler
                if k_switch == True:
                    k1 = k1_default
                    k1_up = k1_up_default 
                if travel_control_l1 == False:
                    s1 = s1_up_limit
                else:
                    s1, s1_up = s_changer_slow(l1, s1, s_modifier1, s1_up, s1_up_limit, s1_down_limit)
                unique_solution_list1.append(solution_space)
                general_unique_solution_list1.append(solution_space)
                expected_l1 += 1
                unique_counter_l1 += 1
    
            # If next layer is disabled, write final L1 results to file
            if next_capacity == 0:
                final_solution_output = convert_numeric_solution(final_solution_output)
                a_solution_space = 0
                for x in final_solution_output:
                    if x == 1:
                        a_solution_space += 1        
                print_solution_to_text_file(final_solution_output, profit_count_solution_space, 
                                            profit_solution_space, text_dimension) #my_dimension_solution_space)
            
            # Per-iteration status/log
            screen_printer(c, layer_no, my_capacity_solution_space, profit_count_solution_space)
            if len(general_unique_solution_list1) <= variety_1 * loop_l1:
                print("Number of unique solutions in layer 1: Expected = ", expected_l1, " Extracted = ", len(general_unique_solution_list1))
                print_log_file("\n" + "Number of unique solutions in layer 1: Expected = " + str(expected_l1) + " Extracted = " + str(len(general_unique_solution_list1)))

            
##################### LAYER 2 #####################

# Run Layer 2 only if enabled (pattern size > 0)
if my_capacity_layer2 > 0:
    l2 = 0                                          # L2-local step counter
    expected_l2 = 0                                 # rolling expected unique count  
    loop_l2 = len(general_unique_solution_list1)    # number of L1 outputs to expand
    
    layer_no = "LAYER 2"
    print()
    print(layer_no)
    print_log_file("\n" + "\n" + layer_no)
    
    # Iterate over each unique solution produced by Layer 1
    for solution_space in general_unique_solution_list1:

        # Reset per-seed trackers
        unique_counter_l2 = 0
        zero_solution_list_2 = []       # for early exit (zero-profit)
        same_solution_list_2 = []       # for early exit (duplicates)
        unique_solution_list2_sS_control = []
        unique_solution_list2_zs_control = []
        s2 = s2_up_limit                # L2 scheduler start (p/s)
        
        # Reset k to defaults (if k oscillation is enabled)
        k2 = k2_default
        k2_up = k2_up_default
        
        # Decode and log incoming solution from Layer 1
        final_solution_output_l1 = convert_numeric_solution(solution_space)
        final_solution_output_l1 = decode_solution(final_solution_output_l1)
        print()
        print("Layer 1 Solution: ", final_solution_output_l1) #solution_space)
        print_log_file("\n" + "\n" +"Layer 1 Solution: " + str(final_solution_output_l1))
        
        unique_solution_list2 = []      # uniques collected at L2 for this seed
        
        # Keep extracting until we reach target variety for this seed (L1 output = seed for L2) 
        while (len(unique_solution_list2) < variety_2):
            l2 += 1
            c += 1      # global step counter
            
            # Reset duplicate tracker when a new unique appears
            if len(unique_solution_list2) > len(unique_solution_list2_sS_control):
                same_solution_list_2 = []
                unique_solution_list2_sS_control.append(unique_solution_list2[-1])
            
            # Reset zero-profit tracker when a new unique appears
            if len(unique_solution_list2) > len(unique_solution_list2_zs_control):
                zero_solution_list_2 = []
                unique_solution_list2_zs_control.append(unique_solution_list2[-1])
            
            # Early exit for this seed if thresholds are hit (zero/duplicate)
            if len(zero_solution_list_2) == eleminate_zero_sol_2 or len(same_solution_list_2) == eleminate_same_sol_2:
                expected_l2 += variety_2 - unique_counter_l2
                if len(general_unique_solution_list2) <= variety_2 * loop_l2:
                    print("Number of unique solutions in layer 2: Expected = ", expected_l2, " Extracted = ", len(general_unique_solution_list2))
                    print_log_file("\n" + "Number of unique solutions in layer 2: Expected = " + str(expected_l2) + " Extracted = " + str(len(general_unique_solution_list2)))
                break
                   
            # Define pattern size for the next layer.
            # If 0 -> no further layer is run, L2 outputs become the final results.
            next_capacity = my_capacity_layer3
            
            # Run L2 (expansion over L1 output with current controls) 
            problem_layer2 = Layered_Multiple_Frequency_Optimization_Layer2(solution_space, profits_original, 
                                                                    weights_original, my_capacity_layer2, my_dimension_layer2, text_dimension,
                                                                    level_of_dim_reduction_l2, zero_one_matrix, random_control_l2, relationship_control_l2)
            
            # Optionally oscillate L2 dimension window before running objective
            if dimension_changer_control_l2 == True:
                d2 = my_dimension_layer2
                d2, d2_up = dimension_changer(d2, d_modifier2, d2_up, d2_up_limit, d2_down_limit)
                my_dimension_layer2 = d2
            
            # Run objective with selected schedule (p or s) and current k
            if p_s == "p":
                (solution_space2, final_solution_output, profit_layer2, cost_layer2, profit_count_layer2, 
                profits_original, weights_original, text_dimension) = problem_layer2.objective_function(p = s2, k = k2) #update 13.12.2024 k = k2) 
            else:
                (solution_space2, final_solution_output, profit_layer2, cost_layer2, profit_count_layer2, 
                profits_original, weights_original, text_dimension) = problem_layer2.objective_function(s = s2, k = k2) #update 13.12.2024 k = k2) 
            
            # Case: zero-profit outcome -> adjust schedulers and continue
            if profit_count_layer2 == 0:
                # First push s/p toward mid-range once, then continue with slow/fast scheduler
                if s2 < int((s2_up_limit + s2_down_limit)/2)-1 or s2 == s2_up_limit:
                    s2 = int((s2_up_limit + s2_down_limit)/2)-1
                    if s2 % 2 == 0:
                        s2 = s2+1
                s2, s2_up = s_changer_slow(l2, s2, s_modifier2, s2_up, s2_up_limit, s2_down_limit)
                zero_solution_list_2.append(solution_space2)
                screen_printer_zero_solution(c, layer_no, my_capacity_layer2, profit_count_layer2)
                # Optionally oscillate k as well (if enabled)
                if k_switch == True:
                    k2_up = True
                    k2, k2_up = s_changer_slow(l2, k2, k_modifier2, k2_up, k2_up_limit, k2_down_limit)
                continue
            else:
                # After a non-zero result, reset k to defaults (if k oscillation is enabled)
                if k_switch == True:
                    k2 = k2_default
                    k2_up = k2_up_default

            # Case: duplicate solution -> adjust schedulers and continue
            if solution_space2 in unique_solution_list2:
                s2, s2_up = s_changer_slow(l2, s2, s_modifier2, s2_up, s2_up_limit, s2_down_limit)
                if k_switch == True:
                    k2, k2_up = s_changer_slow(l2, k2, k_modifier2, k2_up, k2_up_limit, k2_down_limit) 
                same_solution_list_2.append(solution_space2)
                screen_printer_same_solution(c, layer_no, my_capacity_layer2, profit_count_layer2)
                continue
            else:
                # Unique solution -> record and move (or pin) scheduler
                if k_switch == True:
                    k2 = k2_default
                    k2_up = k2_up_default
                if travel_control_l2 == False:
                    s2 = s2_up_limit
                else:
                    s2, s2_up = s_changer_slow(l2, s2, s_modifier2, s2_up, s2_up_limit, s2_down_limit)
                unique_solution_list2.append(solution_space2)
                general_unique_solution_list2.append(solution_space2)
                expected_l2 += 1
                unique_counter_l2 += 1
            
            # If next layer is disabled, write final L2 results to file
            if next_capacity == 0:
                final_solution_output = convert_numeric_solution(final_solution_output)
                a_layer2 = 0
                for x in final_solution_output:
                    if x == 1:
                        a_layer2 += 1
                print_solution_to_text_file(final_solution_output, profit_count_layer2, 
                                            profit_layer2, text_dimension)  
            
            # Per-iteration status/log
            screen_printer(c, layer_no, my_capacity_layer2, profit_count_layer2)
            if len(general_unique_solution_list2) <= variety_2 * loop_l2:
                print("Number of unique solutions in layer 2: Expected = ", expected_l2, " Extracted = ", len(general_unique_solution_list2))
                print_log_file("\n" + "Number of unique solutions in layer 2: Expected = " + str(expected_l2) + " Extracted = " + str(len(general_unique_solution_list2)))
        
        
##################### LAYER 3 #####################

# Run Layer 3 only if enabled (pattern size > 0)   
if my_capacity_layer3 > 0:
    l3 = 0                           # L3-local step counter
    expected_l3 = 0                  # rolling expected unique count
    loop_l3 = len(general_unique_solution_list2)  # number of L2 outputs to expand
   
    layer_no = "LAYER 3"
    print()
    print(layer_no)
    print_log_file("\n" + "\n" + layer_no)
    
    # Iterate over each unique solution produced by Layer 2
    for solution_space2 in general_unique_solution_list2:

        # Reset per-seed trackers
        unique_counter_l3 = 0
        zero_solution_list_3 = []       # for early exit (zero-profit) 
        same_solution_list_3 = []       # for early exit (duplicates)
        unique_solution_list3_sS_control = []
        unique_solution_list3_zs_control = []
        s3 = s3_up_limit                # L3 scheduler start (p/s)
        
        # Reset k to defaults (if k oscillation is enabled)
        k3 = k3_default
        k3_up = k3_up_default
        
        # Decode and log incoming solution from Layer 2
        final_solution_output_l2 = convert_numeric_solution(solution_space2)
        final_solution_output_l2 = decode_solution(final_solution_output_l2)
        print()
        print("Layer 2 Solution: ", final_solution_output_l2) #solution_space2)
        print_log_file("\n" + "\n" +"Layer 2 Solution: " + str(final_solution_output_l2))
        
        unique_solution_list3 = []       # uniques collected at L3 for this seed
        
        # Keep extracting until we reach target variety for this seed (L2 output = seed for L3) 
        while (len(unique_solution_list3) < variety_3):
            l3 += 1
            c += 1      # global step counter
            
            # Reset duplicate tracker when a new unique appears
            if len(unique_solution_list3) > len(unique_solution_list3_sS_control):
                same_solution_list_3 = []
                unique_solution_list3_sS_control.append(unique_solution_list3[-1])
            
            # Reset zero-profit tracker when a new unique appears
            if len(unique_solution_list3) > len(unique_solution_list3_zs_control):
                zero_solution_list_3 = []
                unique_solution_list3_zs_control.append(unique_solution_list3[-1])
            
            # Early exit for this seed if thresholds are hit (zero/duplicate)
            if len(zero_solution_list_3) == eleminate_zero_sol_3 or len(same_solution_list_3) == eleminate_same_sol_3:
                expected_l3 += variety_3 - unique_counter_l3
                if len(general_unique_solution_list3) <= variety_3 * loop_l3:
                    print("Number of unique solutions in layer 3: Expected = ", expected_l3, " Extracted = ", len(general_unique_solution_list3))
                    print_log_file("\n" + "Number of unique solutions in layer 3: Expected = " + str(expected_l3) + " Extracted = " + str(len(general_unique_solution_list3)))
                break        
        
            # Define pattern size for the next layer.
            # If 0 -> no further layer is run, L3 outputs become the final results.
            next_capacity = 0 # my_capacity_layer4
            
            # Run L3 (expansion over L2 output with current controls)   
            problem_layer3 = Layered_Multiple_Frequency_Optimization_Layer3(solution_space2, profits_original, weights_original, 
                                                                    my_capacity_layer3, my_dimension_layer3, text_dimension,
                                                                    level_of_dim_reduction_l3, zero_one_matrix, random_control_l3, relationship_control_l3)
            
            # Optionally oscillate L3 dimension window before running objective
            if dimension_changer_control_l3 == True:
                d3 = my_dimension_layer3
                d3, d3_up = dimension_changer(d3, d_modifier3, d3_up, d3_up_limit, d3_down_limit)
                my_dimension_layer3 = d3
            
            # Run objective with selected schedule (p or s) and current k
            if p_s == "p":
                (solution_space3, final_solution_output, profit_layer3, cost_layer3, profit_count_layer3, 
                profits_original, weights_original, text_dimension) = problem_layer3.objective_function(p = s3, k = k3) #update 13.12.2024 k = k3) 
            else:
                (solution_space3, final_solution_output, profit_layer3, cost_layer3, profit_count_layer3, 
                profits_original, weights_original, text_dimension) = problem_layer3.objective_function(s = s3, k = k3) #update 13.12.2024 k = k3) 
            
            # Case: zero-profit outcome -> adjust schedulers and continue
            if profit_count_layer3 == 0:
                # First push s/p toward mid-range once, then continue with slow/fast scheduler
                if s3 < int((s3_up_limit + s3_down_limit)/2)-1 or s3 == s3_up_limit:
                    s3 = int((s3_up_limit + s3_down_limit)/2)-1
                    if s3 % 2 == 0:
                        s3 = s3+1
                s3, s3_up = s_changer_slow(l3, s3, s_modifier3, s3_up, s3_up_limit, s3_down_limit)
                zero_solution_list_3.append(solution_space3)
                screen_printer_zero_solution(c, layer_no, my_capacity_layer3, profit_count_layer3)
                # Optionally oscillate k as well (if enabled)
                if k_switch == True:
                    k3_up = True
                    k3, k3_up = s_changer_slow(l3, k3, k_modifier3, k3_up, k3_up_limit, k3_down_limit)
                continue
            else:
                # After a non-zero result, reset k to defaults (if k oscillation is enabled)
                if k_switch == True:
                    k3 = k3_default
                    k3_up = k3_up_default
   
           # Case: duplicate solution -> adjust schedulers and continue
            if solution_space3 in unique_solution_list3:
                s3, s3_up = s_changer_slow(l3, s3, s_modifier3, s3_up, s3_up_limit, s3_down_limit)
                if k_switch == True:
                    k3, k3_up = s_changer_slow(l3, k3, k_modifier3, k3_up, k3_up_limit, k3_down_limit)    
                same_solution_list_3.append(solution_space3)
                screen_printer_same_solution(c, layer_no, my_capacity_layer3, profit_count_layer3)
                continue
            else:
                # Unique solution -> record and move (or pin) scheduler
                if k_switch == True:
                    k3 = k3_default
                    k3_up = k3_up_default
                if travel_control_l3 == False:
                    s3 = s3_up_limit
                else:
                    s3, s3_up = s_changer_slow(l3, s3, s_modifier3, s3_up, s3_up_limit, s3_down_limit)
                unique_solution_list3.append(solution_space3)
                general_unique_solution_list3.append(solution_space3)
                expected_l3 += 1
                unique_counter_l3 += 1
                
            # If next layer is disabled, write final L3 results to file)
            if next_capacity == 0:
                final_solution_output = convert_numeric_solution(final_solution_output)
                a_layer3 = 0
                for x in final_solution_output:
                    if x == 1:
                        a_layer3 += 1
                print_solution_to_text_file(final_solution_output, profit_count_layer3, 
                                            profit_layer3, text_dimension) 
            
            # Per-iteration status/log
            screen_printer(c, layer_no, my_capacity_layer3, profit_count_layer3)
            if len(general_unique_solution_list3) <= variety_3 * loop_l3:
                print("Number of unique solutions in layer 3: Expected = ", expected_l3, " Extracted = ", len(general_unique_solution_list3))
                print_log_file("\n" + "Number of unique solutions in layer 3: Expected = " + str(expected_l3) + " Extracted = " + str(len(general_unique_solution_list3)))
                    
                
# Close files (run end)
f.close()       # close Solution.txt
f_log.close()   # close Log.txt

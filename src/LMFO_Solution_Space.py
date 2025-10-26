# -*- coding: utf-8 -*-
# -------------------------------------
# LMFO – Layer 1 (Solution Space)
# File: LMFO_Solution_Space.py
# Author: Ishak Kutlu
# Created: 2024-04-28
#
# Executive summary
# -------------------------------------
# Layer 1 turns noisy, limited data into high-value patterns:
# • Loads per-item vectors, applies robust scaling (mean or mean/sqrt(std)).
# • Injects cross-run “signal blocks” to pre-bias features by co-activation.
# • Orders features (global or relationship-aware) -> phase-1 reduction to a workable window.
# • Balances representation via ROS/SMOTE and re-orders -> phase-2 reduction to the final L1 window.
# • Runs an adaptive search (leader + arms) to extract the best pattern.
# • Maps results back to original indexing in two steps (augmented -> L1 view -> original order) for upper layers.
#
# Key flags
# -------------------------------------
# • relationship_control_l1 : True -> relationship-aware (co-activation) ordering;      False -> global ordering.
# • feed_list_control       : True -> apply cross-run pre-bias from signal blocks;      False -> skip.
# • avg_sqrt_std_control    : True -> scale by mean/sqrt(std);                          False -> mean only.
# • random_control_l1       : True -> protect pinned (fixed) indices from random flips; False -> freer edits.
# • fixed_solution_list     : Non-empty -> enable seeding (pin listed indices as True); empty -> no seeding.
# • level_of_dim_reduction_l1: Second-stage reduction dimension after data augmentation.
#
# Inputs/Outputs
# -------------------------------------
# In :
# • folderName / fileName (training data)
# • my_dimension_solution_space (phase-1 reduction dimension)
# • level_of_dim_reduction_l1 (phase-2 reduction dimension)
# • fixed_solution_list (signal blocks)
# • feed_list_control
# • relationship_control_l1
# • random_control_l1
# • avg_sqrt_std_control (scaling mode: mean vs. mean/sqrt(std))
#
# Out:
# • final_solution      (pass-through) & (mapped)
# • final_solution_output
# • final_profit
# • final_cost
# • profit_count
# • profits_original    (pass-through)
# • weights_original    (pass-through)
# • text_dimension      (pass-through)
# • zero_one_matrix     (pass-through)
#
# Traceability (main functions)
# -------------------------------------
# • convert_data / convert_data_and_get_sort_parameters   -> per-item scaling & importance scores
# • solution_space_of_selected_items                      -> relationship-aware projection (co-activation)
# • create_major_solution_matrix                          -> Expand boolean mask to per-sample matrix; backbone 
#                                                            for profit scoring & relationship projection
# • sort_profits_and_solutions / sort_data                -> ordering + index maps
# • augmented_data / augmented_data_no_constant           -> ROS/SMOTE pipelines
# • objective_function                                    -> leader/arms search under capacity constraint
# • profit_function                                       -> scoring (hit-weighted co-activation)
# • cost_function                                         -> capacity cost (cardinality; unit weights)
# • rearrange_solutions_for_decoding                      -> reverse index mapping (augmented -> L1 view -> original order)
# • helpers: valid_solution/repair/optimizing_stage/neighbor_solution, sorting.
#
# Notes
# -------------------------------------
# • Class labels for ROS/SMOTE are derived from positive-count per row.
# • Profit metric multiplies average co-activation payoff by support count; this is intentional
#   to reward stable, repeated co-activations across samples.


from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
import random
import statistics
import math
import copy

class Layered_Multiple_Frequency_Optimization_Solution_Space:
    def __init__(self, folderName, fileName, my_dimension_solution_space, fixed_solution_list, level_of_dim_reduction_l1, feed_list, feed_list_control,
                 random_control_l1, relationship_control_l1, avg_sqrt_std_control):
        
        # -------------------------------------------------------------------------
        # L1 (Solution Space) setup: core config & flags
        # -------------------------------------------------------------------------
        
        self.my_dimension_solution_space = my_dimension_solution_space
        self.folderName = folderName
        self.fileName = fileName      
        self.weights = []
        self.profits = []
        self.init_solutions = []
        self.profit_count = 0
        self.random_control = random_control_l1  # controls random edits; True -> protect pinned items from random flips, False -> allow edits
        self.fixed_solution_list = fixed_solution_list      # When enabled, iterate over fixed seeds as initial solutions
        self.level_of_dim_reduction_l1 = level_of_dim_reduction_l1
        self.feed_list = feed_list                          # signal blocks
        self.feed_list_control = feed_list_control          # if True, reweight the global profit matrix derived from training data 
        feed_control = False
        self.feed_control = feed_control
        self.relationship_control = relationship_control_l1 # if True, enforce relational constraints; if False, ignore them        
        # scaling mode: avg/sqrt(std) vs avg-only
        self.avg_sqrt_std_control = avg_sqrt_std_control    # True -> use mean/sqrt(std) weighting, False -> use mean only


        # -------------------------------------------------------------------------
        # Load training data from text file (<dimension> <capacity>, per-item profit vectors, weights)
        # -------------------------------------------------------------------------
        
        with open(f"{folderName}/{fileName}") as f:           
            self.dimension, self.capacity = map(int, f.readline().split())
            for i in range(self.dimension):
                p_i, w_i = map(str, f.readline().split('_ '))
                w_i = w_i.replace(",", ".")
                p_i = p_i.replace(",", ".")
                self.weights.append(float(w_i))
                p_i = map(float, p_i.split())
                self.profits.append(list(p_i))
        
        # Keep a copy of original text dimension (for decoding / reporting)
        self.text_dimension =  copy.deepcopy(self.dimension)
        
        
        # -------------------------------------------------------------------------
        # Initial conversion: scale profit matrix (avg/sqrt(std) or avg-only) & stash originals
        # -------------------------------------------------------------------------
        
        zero_one_matrix = copy.deepcopy(self.profits)     
        self.zero_one_matrix = copy.deepcopy(self.profits) 
        converted_profits = self.convert_data(self.profits)
        self.profits = converted_profits
        self.profits_original = copy.deepcopy(self.profits)
        self.weights_original = copy.deepcopy(self.weights)


        # -------------------------------------------------------------------------
        # Fixed seed mask (pin selected indices to True in the search space)
        # -------------------------------------------------------------------------
        
        fixed_solution = []
        if len(fixed_solution_list) > 0:
            fixed_solution =  [False for j in range(self.dimension)]
            for fixed_item in self.fixed_solution_list:
                fixed_solution[fixed_item-1] = True
        self.fixed_solution = copy.deepcopy(fixed_solution)  
        
        
        # -------------------------------------------------------------------------
        # External feed list (run-to-run memory) 
        # Reweight the global profit matrix once using prior signal blocks;
        # co-activation pre-bias -> propagated to upper layers.
        # -------------------------------------------------------------------------
        
        # For each signal block, compute a co-activation–scaled profit view
        # and accumulate into a pre-biased matrix.
        if self.feed_list_control == True:
            self.avg_control = False
            feed_control = True                                     # disable sorting inside helper
            self.feed_control = feed_control
            stored_feed_profits = copy.deepcopy(self.profits)
            stored_feed_profits = np.dot(stored_feed_profits, 0)    # zero-like accumulator
            k = 0
            while k < len(self.feed_list):
                feed_profits = copy.deepcopy(self.profits)
                feed_weights = copy.deepcopy(self.weights)
                feed_solution_list = self.feed_list[k]
                k += 1
                # Build 1-based mask of the given signal block (pin these indices)
                # Note: Indices are 1-based because they represent feature IDs (1–80).
                feed_solution =  [False for j in range(self.dimension)]
                for fixed_item in feed_solution_list:
                    feed_solution[fixed_item-1] = True
                # Project around the signal block and scale rows by co-activation; no sorting when feed_control=True
                feed_solution, feed_profits, feed_weights, sorting_index_list = self.solution_space_of_selected_items(self.dimension, feed_solution, feed_profits, feed_weights)
                stored_feed_profits += feed_profits
            # Replace profits with redistributed view and restore normal flow
            self.profits = stored_feed_profits  
            feed_control = False
            self.feed_control = feed_control
            self.profits_original = copy.deepcopy( self.profits)
            self.weights_original = copy.deepcopy(self.weights)


        # -------------------------------------------------------------------------
        # Relationship-aware sort (if any fixed items) OR plain sort otherwise
        #   True  -> co-activation local ordering around fixed items (relationship-aware)
        #   False -> plain/global ordering by avg/sqrt(std) (no relationship model)
        # -------------------------------------------------------------------------
        
        if len(fixed_solution) > 0:
            self.avg_control = False
            if self.relationship_control == False:
                # Plain/global ordering:
                # - Derive per-item sort parameters from the raw copy (zero_one_matrix)
                #   to avoid compounding prior transforms.
                # - Globally sort features; remap fixed items within that order.
                converted_profits, sort_list = self.convert_data_and_get_sort_parameters(zero_one_matrix)              
                fixed_solution, profits, weights, sorting_index_list_l1 = self.sort_profits_and_solutions(self.dimension, fixed_solution, sort_list, converted_profits, self.weights)
            else:
                # Relationship-aware (co-activation local ordering):
                # - Build a localized subspace around the fixed items using conditional co-activation.
                # - Return re-ordered profits/weights and the mapping index list for Layer 1.
                fixed_solution, profits, weights, sorting_index_list_l1 = self.solution_space_of_selected_items(self.dimension, fixed_solution, self.profits, self.weights)
            # Commit the working view and keep the index mapping for later decoding
            self.fixed_solution = copy.deepcopy(fixed_solution)
            self.profits = copy.deepcopy(profits)
            self.weights = copy.deepcopy(weights)
            self.sorting_index_list_l1 = copy.deepcopy(sorting_index_list_l1)
        else:
            # No fixed items present -> plain/global ordering for the full set
            converted_profits, sort_parameters_list = self.convert_data_and_get_sort_parameters(zero_one_matrix)
            sorting_profits, sorting_weights, sorting_index_list_l1 = self.sort_data(converted_profits, self.weights, sort_parameters_list)
            self.profits = sorting_profits
            self.weights = sorting_weights
            self.sorting_index_list_l1 = sorting_index_list_l1


        # -------------------------------------------------------------------------
        # Phase-1 dimension reduction: initial working window for Layer 1
        # Limit the working arrays to the top-d items selected/ordered above.
        # -------------------------------------------------------------------------
        
        # Deepcopy to avoid overwriting upstream data
        self.dimension = copy.deepcopy(self.my_dimension_solution_space)
        self.dimension_first_phase = copy.deepcopy(self.my_dimension_solution_space)
        d = self.dimension
        # Truncate working views to the first d features
        self.profits = self.profits[:d]
        self.weights = self.weights[:d]
        self.sorting_index_list_l1 = self.sorting_index_list_l1[:d]
        # If a fixed-seed mask exists, truncate it to the same window
        if len(fixed_solution) > 0:  
            self.fixed_solution = self.fixed_solution[:d]
        
        
        # -------------------------------------------------------------------------
        # Data augmentation (ROS/SMOTE), then re-order
        # If fixed seeds exist & relationship_control=True -> relationship-aware augmented pipeline;
        # otherwise use the plain augmented pipeline.
        # -------------------------------------------------------------------------
        if len(fixed_solution) > 0:
            # Relationship-aware augmented pipeline (fixed seeds present)
            self.avg_control = True            
            augmented_fixed_solution, augmented_profits, augmented_weights, sorting_augmented_index_list_l1 = self.augmented_data()
            self.fixed_solution = augmented_fixed_solution
            self.profits = augmented_profits
            self.weights = augmented_weights
            self.sorting_augmented_index_list_l1 = sorting_augmented_index_list_l1
        else:
            # Plain augmented pipeline (no fixed seeds)
            augmented_profits, augmented_weights, sorting_augmented_index_list_l1 = self.augmented_data_no_constant()
            self.profits = augmented_profits
            self.weights = augmented_weights
            self.sorting_augmented_index_list_l1 = sorting_augmented_index_list_l1
        
        
        # -------------------------------------------------------------------------
        # Phase-2 dimension reduction: final L1 window after augmentation
        # Align all working arrays (and mask, if any) to the final size d.
        # -------------------------------------------------------------------------
        
        self.dimension = copy.deepcopy(self.level_of_dim_reduction_l1)
        d = self.dimension
        self.profits = self.profits[:d]
        self.weights = self.weights[:d]
        self.sorting_augmented_index_list_l1 = self.sorting_augmented_index_list_l1[:d] 
        if len(fixed_solution) > 0:  
            self.fixed_solution = self.fixed_solution[:d]

        
    # -------------------------------------------------------------------------
    # Objective: adaptive search under capacity (leader + arms, k-neighbors)
    # -------------------------------------------------------------------------

    # NOTE on hyperparameters:
    # These values are passed by RUN_LMFO at call time.
    # The defaults below are placeholders only.
    #
    # s (from RUN file): total neighbor-evaluation capacity constraint
    # p (from RUN file): population size (odd: 1 leader + two arms)
    # k (from RUN file): neighbors per candidate
    # m: micro-epochs per step [local]
    # x: top neighbors shared from leader [local]
    def objective_function(self, s = 20, p = 1, k = 5, m = 1, x = 0): 
        d = self.dimension
        
        # Generate p random candidates and split them:
        init_solutions = self.initial_solutions(p, d)
        leader_solution = init_solutions[0]
        left_list = init_solutions[1:int(((p-1)/2)+1)]
        right_list = init_solutions[int(((p-1)/2)+1):p]
        
        # Make initial candidates capacity-valid once
        left_list_valid = []
        right_list_valid = []
        leader_solution_valid = self.valid_solution(leader_solution)
        for val_sol_left, val_sol_right in zip(left_list, right_list):
            val_sol_left = self.valid_solution(val_sol_left)
            val_sol_right = self.valid_solution(val_sol_right)
            left_list_valid.append(val_sol_left)
            right_list_valid.append(val_sol_right)
        
        # Main loop: rotate leader across arms while consuming the step capacity constraint (s)
        left_side = True
        i = 0
        while i < s:
            # m micro-epochs: explore neighbors for leader and arms, then update bests
            for j in range(m):
                # Leader exploration: build k neighbors, keep the best
                neighbor_leader_list = []
                a = 0 
                for a in range(k):
                    neighbor_leader = copy.deepcopy(leader_solution_valid)
                    neighbor_leader_list.append(self.neighbor_solution(neighbor_leader))
                neighbor_leader_list = self.sorting(neighbor_leader_list)
                i += k
                if self.profit_function(leader_solution_valid) < self.profit_function(neighbor_leader_list[0]):
                    leader_solution_valid = neighbor_leader_list[0]
                # Optionally share top-x leader neighbors to arms (diversify arms with leader hints)
                share_count = (k-1)/2
                neighbor_left_list = []
                neighbor_right_list = []
                if 1 <= x and x <= share_count:
                    a = 1
                    y = 0
                    while y < x:
                        neighbor_left_list.append(neighbor_leader_list[a])
                        y += 1
                        a += 2
                    a = 2
                    y = 0
                    while y < x:
                        neighbor_right_list.append(neighbor_leader_list[a])
                        y += 1
                        a += 2
                
                # Arms exploration: for each arm slot, try (k-x) fresh neighbors and keep the best
                t = 1
                while t < (p-1)/2:
                    for a in range(k-x):
                        neighbor_left = copy.deepcopy(left_list_valid[t])
                        neighbor_left_list.append(self.neighbor_solution(neighbor_left))
                    neighbor_left_list = self.sorting(neighbor_left_list)
                    for a in range(k-x):
                        neighbor_right = copy.deepcopy(right_list_valid[t])
                        neighbor_right_list.append(self.neighbor_solution(neighbor_right))
                    neighbor_right_list = self.sorting(neighbor_right_list)
                    i += 2*(k - x) 
                    # Commit arm improvements if a better neighbor is found
                    if self.profit_function(left_list_valid[t]) < self.profit_function(neighbor_left_list[0]):
                        left_list_valid[t] = neighbor_left_list[0]
                    if self.profit_function(right_list_valid[t]) < self.profit_function(neighbor_right_list[0]):
                        right_list_valid[t] = neighbor_right_list[0]
                    # Prepare next round’s share lists (if sharing is enabled)
                    neighbor_left_list_next = []
                    neighbor_right_list_next = []
                    if 1 <= x and x <= share_count:
                        a = 1
                        y = 0
                        while y < x:
                            neighbor_left_list_next.append(neighbor_left_list[a])
                            neighbor_right_list_next.append(neighbor_right_list[a])
                            y += 1
                            a += 1
                    neighbor_left_list = copy.deepcopy(neighbor_left_list_next)
                    neighbor_right_list = copy.deepcopy(neighbor_right_list_next)
                    t+=1
            
            # Leader rotation (queue-style):
            # cycle leader across left/right arms to spread improvements   
            if left_side == True:
                left_list_valid.append(leader_solution_valid)
                leader_solution_valid = copy.deepcopy(left_list_valid[0])
                del left_list_valid[0]
                left_side = False
            else:
                right_list_valid.append(leader_solution_valid)
                leader_solution_valid = copy.deepcopy(right_list_valid[0])
                del right_list_valid[0]
                left_side = True   
        
        # Gather all candidates, sort, pick the best final solution
        final_list_solution = []
        final_list_solution = copy.deepcopy(left_list_valid)
        final_list_solution.append(leader_solution_valid)
        for item in right_list_valid:
            final_list_solution.append(item)
        final_list_solution = self.sorting(final_list_solution)
        final_solution = final_list_solution[0]
        final_profit = self.profit_function(final_solution)        
        final_cost = self.cost_function(final_solution)
        
        # Map back from augmented and Phase-1 orderings to original text order
        text_dimension = copy.deepcopy(self.text_dimension)
        final_solution_output = copy.deepcopy(final_solution)

        sorting_augmented_index_list_l1 = copy.deepcopy(self.sorting_augmented_index_list_l1)
        final_solution_output = self.rearrange_solutions_for_decoding(final_solution_output, sorting_augmented_index_list_l1, self.dimension_first_phase)
        final_solution = self.rearrange_solutions_for_decoding(final_solution, sorting_augmented_index_list_l1, self.dimension_first_phase)
        
        sorting_index_list_l1 = copy.deepcopy(self.sorting_index_list_l1)
        final_solution_output = self.rearrange_solutions_for_decoding(final_solution_output, sorting_index_list_l1, text_dimension)
        final_solution = self.rearrange_solutions_for_decoding(final_solution, sorting_index_list_l1, text_dimension)
        
        # Return best pattern + metrics + originals for upper layers
        weights_original = copy.deepcopy(self.weights_original)
        profits_original =  copy.deepcopy(self.profits_original)
        zero_one_matrix =  copy.deepcopy(self.zero_one_matrix)
        
        return (final_solution, final_solution_output, final_profit, final_cost, self.profit_count,
                profits_original, weights_original, text_dimension, zero_one_matrix)
    

    # -------------------------------------------------------------------------
    # # Reverse index mapping: remap items to target order using sorting_index_list.
    # -------------------------------------------------------------------------
    
    def rearrange_solutions_for_decoding (self, final_solution, sorting_index_list, d):
        rearrange_solution = [False] * d 
        i=0
        for item in final_solution:
            rearrange_solution[sorting_index_list[i]] = final_solution[i]
            i+=1
        return rearrange_solution
    

    # -------------------------------------------------------------------------
    # Relationship projection & scaling (conditional co-activation)
    # - Build mask from fixed items in final_solution.
    # - For each feature i: count samples where i AND all fixed items ≠ 0,
    #   compute my_avg = count / N, normalize my_avg by the first nonzero entry in row i, then rescale the entire row by this ratio.
    # - In feed mode, sorting is skipped; otherwise sort by avg and return mapping.
    # -------------------------------------------------------------------------

    def solution_space_of_selected_items(self, dim_next, final_solution, profit_matrix, weight_vector):
        d = self.dimension
        number_of_sample = int(sum(map(len, profit_matrix))/d) 
        # Fixed indices (True positions) and their per-sample mask
        true_list = [a for a, x in enumerate(final_solution) if x] 
        n_true_list = len(true_list)
        major_solution = []
        major_solution = self.create_major_solution_matrix(final_solution) 
        avg_list = []
        for i in range(d):
            my_counter = 0.0
            profit = 0.0
            for j in range(number_of_sample):
                # Conditional co-activation: i counts only if all fixed positions are non-zero.
                # If i itself is fixed, give full credit (profit = N) to keep it prioritized.
                if n_true_list == 1:
                    if i != true_list[0]:
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 2:
                    if i != true_list[0] and i != true_list[1]:
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 3:
                    if i != true_list[0] and i != true_list[1] and i != true_list[2]:
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 4:
                    if (
                            i != true_list[0] and 
                            i != true_list[1] and 
                            i != true_list[2] and 
                            i != true_list[3]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1     
                    else:
                        profit = number_of_sample  
                elif n_true_list == 5:
                    if (
                            i != true_list[0] and 
                            i != true_list[1] and 
                            i != true_list[2] and 
                            i != true_list[3] and 
                            i != true_list[4]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 6:
                    if (
                            i != true_list[0] and
                            i != true_list[1] and
                            i != true_list[2] and
                            i != true_list[3] and
                            i != true_list[4] and
                            i != true_list[5]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0 and
                            major_solution[true_list[5]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 7:
                    if (
                            i != true_list[0] and
                            i != true_list[1] and
                            i != true_list[2] and
                            i != true_list[3] and
                            i != true_list[4] and
                            i != true_list[5] and
                            i != true_list[6]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0 and
                            major_solution[true_list[5]][j] != 0 and
                            major_solution[true_list[6]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 8:
                    if (
                            i != true_list[0] and
                            i != true_list[1] and
                            i != true_list[2] and
                            i != true_list[3] and
                            i != true_list[4] and
                            i != true_list[5] and
                            i != true_list[6] and
                            i != true_list[7]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0 and
                            major_solution[true_list[5]][j] != 0 and
                            major_solution[true_list[6]][j] != 0 and
                            major_solution[true_list[7]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 9:
                    if (
                            i != true_list[0] and
                            i != true_list[1] and
                            i != true_list[2] and
                            i != true_list[3] and
                            i != true_list[4] and
                            i != true_list[5] and
                            i != true_list[6] and
                            i != true_list[7] and
                            i != true_list[8]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0 and
                            major_solution[true_list[5]][j] != 0 and
                            major_solution[true_list[6]][j] != 0 and
                            major_solution[true_list[7]][j] != 0 and
                            major_solution[true_list[8]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
                elif n_true_list == 10:
                    if (
                            i != true_list[0] and
                            i != true_list[1] and
                            i != true_list[2] and
                            i != true_list[3] and
                            i != true_list[4] and
                            i != true_list[5] and
                            i != true_list[6] and
                            i != true_list[7] and
                            i != true_list[8] and
                            i != true_list[9]
                            ):
                        if (
                            profit_matrix[i][j] != 0 and 
                            major_solution[true_list[0]][j] != 0 and 
                            major_solution[true_list[1]][j] != 0 and
                            major_solution[true_list[2]][j] != 0 and
                            major_solution[true_list[3]][j] != 0 and
                            major_solution[true_list[4]][j] != 0 and
                            major_solution[true_list[5]][j] != 0 and
                            major_solution[true_list[6]][j] != 0 and
                            major_solution[true_list[7]][j] != 0 and
                            major_solution[true_list[8]][j] != 0 and
                            major_solution[true_list[9]][j] != 0
                            ):
                            profit += profit_matrix[i][j]
                            my_counter += 1
                    else:
                        profit = number_of_sample
            
            # Average support over N, then normalize by the first positive entry and scale row
            my_avg = profit / number_of_sample
            avg_list.append(my_avg)
            my_val = 0
            for my_val in profit_matrix[i][:]:
                if my_val > 0:
                    break
            if my_val > 0:
                my_avg = my_avg / my_val
            profit_matrix[i] = np.dot(profit_matrix[i], my_avg)

        # Sort by avg (desc) and return mapping unless feed mode is active
        if self.feed_control == False:
            final_solution, profit_matrix, weight_vector, sorting_index_list = self.sort_profits_and_solutions(dim_next, final_solution, avg_list, profit_matrix, weight_vector)
        else:
            sorting_index_list = []
        return final_solution, profit_matrix, weight_vector, sorting_index_list
    

    # -------------------------------------------------------------------------
    # Ordering by importance score (avg_list)
    # - If relationship_control == False (plain), boost fixed items (set score=1) so they stay on top.
    # - Build sorting_index_list by descending avg_list, then reorder:
    #   final_solution, profit_matrix (rows), and weight_vector accordingly.
    # - Return also sorting_index_list for later reverse mapping.
    # -------------------------------------------------------------------------

    def sort_profits_and_solutions (self, d, final_solution, avg_list, profit_matrix, weight_vector):
        # Plain mode: pin fixed items at the top of the order
        if self.relationship_control == False:
            true_list = [a for a, x in enumerate(final_solution) if x]
            for i in range(d):
                for fixed_element in true_list:
                    if i == fixed_element:
                        avg_list[i] = 1
        # Greedy argmax to produce a descending index order; mark used entries
        sorting_index_list = []
        for item in range(len(final_solution)):
            sort_index = np.argmax(avg_list)
            sorting_index_list.append(sort_index)
            avg_list[sort_index] = -1
        # Apply the same order to all aligned arrays
        sorting_solution = []
        sorting_profits = []
        sorting_weights = []
        for item in sorting_index_list:
            sorting_solution.append(final_solution[item])
            sorting_profits.append(profit_matrix[item])
            sorting_weights.append(weight_vector[item])
        return sorting_solution, sorting_profits, sorting_weights, sorting_index_list
    

    # -------------------------------------------------------------------------
    # Expand minor (boolean) solution to a per-sample “major” matrix
    # - minor_solution: length-d boolean mask (selected items = True)
    # - For each feature i:
    #     • if not selected -> a length-N vector of False
    #     • if selected     -> the per-sample profit row for i (scaled values)
    # - Used later for conditional co-activation checks (non-zero across pinned items).
    # -------------------------------------------------------------------------

    def create_major_solution_matrix(self, minor_solution):
        d = self.dimension
        minor_solution = list(minor_solution)
        major_solution = []
        number_of_sample = int(sum(map(len, self.profits))/d)
        major_solution = []
        for i in range(d):
            if minor_solution[i] == False:
                false_defined_solution_vector =  [False for j in range(number_of_sample)]
                major_solution.append(false_defined_solution_vector)
            else:
                a = minor_solution[i]*self.profits[i]
                major_solution.append(list(a))
        return major_solution
    
    
    def array_converter(self, convert_list_to_array):
        convert_list_to_array = np.array(convert_list_to_array)
        return convert_list_to_array


    # -------------------------------------------------------------------------
    # Scoring — profit (hit-weighted co-activation)
    # - Build per-sample “major” matrix from the boolean pattern.
    # - For each sample x: if all selected items are non-zero at x,
    #   sum their values and increment a hit counter.
    # - Final score: profit = (hit_count * accumulated_sum) / N  (N = #samples).
    # - Assumes 1 ≤ n_true_list ≤ 10 (explicit branches below).
    # -------------------------------------------------------------------------

    def profit_function(self, solution):
        major_solution = self.create_major_solution_matrix(solution)  # per-sample view
        true_list = [a for a, x in enumerate(solution) if x]
        n_true_list = len(true_list)
        profit = 0.000000
        profit = float(profit)
        self.profit_count = 0
        for x in range(len(major_solution[0])):     # loop as number of sample
            if n_true_list == 1:
                if (
                    major_solution[true_list[0]][x] != 0 
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 2:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 3:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 4:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 5:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 6:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0 and
                    major_solution[true_list[5]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 7:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0 and
                    major_solution[true_list[5]][x] != 0 and
                    major_solution[true_list[6]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 8:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0 and
                    major_solution[true_list[5]][x] != 0 and
                    major_solution[true_list[6]][x] != 0 and
                    major_solution[true_list[7]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 9:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0 and
                    major_solution[true_list[5]][x] != 0 and
                    major_solution[true_list[6]][x] != 0 and
                    major_solution[true_list[7]][x] != 0 and
                    major_solution[true_list[8]][x] != 0
                    ):
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1
            elif n_true_list == 10:
                if (
                    major_solution[true_list[0]][x] != 0 and 
                    major_solution[true_list[1]][x] != 0 and
                    major_solution[true_list[2]][x] != 0 and
                    major_solution[true_list[3]][x] != 0 and
                    major_solution[true_list[4]][x] != 0 and
                    major_solution[true_list[5]][x] != 0 and
                    major_solution[true_list[6]][x] != 0 and
                    major_solution[true_list[7]][x] != 0 and
                    major_solution[true_list[8]][x] != 0 and
                    major_solution[true_list[9]][x] != 0
                    ):                
                    for indx in range(n_true_list): profit += major_solution[true_list[indx]][x]
                    self.profit_count += 1

        number_of_sample = int(sum(map(len, self.profits))/self.dimension)
        profit = (self.profit_count * profit ) / number_of_sample
        return profit
        

    # -------------------------------------------------------------------------
    # Cost — cardinality (weights are all ones; dummy placeholder)
    # NOTE: self.weights ≡ 1-vector -> cost == number of selected items.
    # Capacity check thus limits max pattern size; ranking is driven by profit.
    # -------------------------------------------------------------------------

    def cost_function(self, solution): 
        weights_arr = self.weights
        solution = self.array_converter(solution)
        weights_arr = self.array_converter(weights_arr)
        cost = np.vdot(weights_arr, solution)
        return cost
    
 
    # -------------------------------------------------------------------------
    # Neighborhood move (1-out, up-to-1-in) under capacity
    # - Respect random_control/random_index_control (e.g., pinned items not flipped).
    # - Remove one random True, then try adding one random False if it fits capacity.
    # - Single addition only (local, small-step exploration).
    # -------------------------------------------------------------------------

    def neighbor_solution(self, solution):
        # pick randomly a removable index currently True (respect constraints)
        random_control = copy.deepcopy(self.random_control)
        remove_index = random.choice(range(len(solution)))
        random_control = self.random_index_control(remove_index)
        while random_control == False:
            remove_index = random.choice(range(len(solution)))
            random_control = self.random_index_control(remove_index)
        sel_value = solution[remove_index]
        while sel_value == False:
            remove_index = random.choice(range(len(solution)))
            random_control = self.random_index_control(remove_index)
            while random_control == False:
                remove_index = random.choice(range(len(solution)))
                random_control = self.random_index_control(remove_index)
            sel_value = solution[remove_index]
        solution[remove_index] = False
        
        # try to add one feasible new index if capacity allows
        cap_val = self.cost_function(solution)
        add_index = random.choice(range(len(solution)))
        sel_value = solution[add_index]
        while sel_value == True:
            add_index = random.choice(range(len(solution)))
            sel_value = solution[add_index]
        while cap_val + self.weights[add_index] <= self.capacity:
            solution[add_index] = True
            cap_val += self.weights[add_index]
            break   # only one addition
        return solution


    # -------------------------------------------------------------------------
    # Feasibility + quick improve
    # - Enforce fixed elements, repair if overweight, then run optimizing_stage;
    #   otherwise directly optimizing_stage (fill remaining capacity).
    # -------------------------------------------------------------------------

    def valid_solution(self, solution):
        # guarantee pinned items are set
        solution = self.fixed_elements(solution)
        cap_val = self.cost_function(solution)
        if cap_val > self.capacity:
            solution = self.repair(solution)            # drop worst until <= capacity
            solution = self.optimizing_stage(solution)  # then greedily fill
        else:
            solution = self.optimizing_stage(solution)
        return solution


    # -------------------------------------------------------------------------
    # Repair — shrink to capacity (random drop, respects constraints)
    # - While cost > capacity: pick a removable True index and unset it.
    # - Respects random_index_control (e.g., do not drop pinned items).
    # - Unit weights -> cost is cardinality; this removes by count.
    # -------------------------------------------------------------------------

    def repair(self, solution):
        cap_val = self.cost_function(solution)
        random_control = copy.deepcopy(self.random_control)
        while cap_val > self.capacity:
            remove_index = random.choice(range(len(solution)))
            random_control = self.random_index_control(remove_index)
            while random_control == False:
                remove_index = random.choice(range(len(solution)))
                random_control = self.random_index_control(remove_index)
            sel_value = solution[remove_index]
            while sel_value == False:
                remove_index = random.choice(range(len(solution)))
                random_control = self.random_index_control(remove_index)
                while random_control == False:
                    remove_index = random.choice(range(len(solution)))
                    random_control = self.random_index_control(remove_index)
                sel_value = solution[remove_index]
            solution[remove_index] = False
            cap_val -= self.weights[remove_index]
        return solution


    # -------------------------------------------------------------------------
    # Optimizing stage — fill up to capacity (random add)
    # - Repeatedly pick a False index; set True if it fits capacity.
    # - Continues adding until full (no single-step break); keeps diversity higher.
    # - Unit weights -> capacity is a max-size cap; profit drives ranking.
    # -------------------------------------------------------------------------

    def optimizing_stage(self, solution):
        cap_val = self.cost_function(solution)
        add_index = random.choice(range(len(solution)))
        sel_value = solution[add_index]
        while sel_value == True:
            add_index = random.choice(range(len(solution)))
            sel_value = solution[add_index]
        while cap_val + self.weights[add_index] <= self.capacity:
            solution[add_index] = True
            cap_val += self.weights[add_index]
            # break     # Early stop; reduces exploration/diversity.
            add_index = random.choice(range(len(solution)))
            sel_value = solution[add_index]
            while sel_value == True:
                add_index = random.choice(range(len(solution)))
                sel_value = solution[add_index]
        return solution


    # -------------------------------------------------------------------------
    # Rank neighbors by profit (greedy argmax, descending)
    # -------------------------------------------------------------------------

    def sorting(self, neighbor_list):
        sorting_list = []
        profit_list = []
        for item in neighbor_list:
            item_profit = self.profit_function(item)
            profit_list.append(item_profit)
        for item in neighbor_list:
            sort_index = np.argmax(profit_list)
            sorting_list.append(sort_index)
            profit_list[sort_index] = -1
        sorting_solution_list = []
        for item in sorting_list:
            sorting_solution_list.append(neighbor_list[item])
        return sorting_solution_list    

    # -------------------------------------------------------------------------
    # Generate p random initial candidates (boolean length d) and append to pool
    # Appends to self.init_solutions (stateful within this instance).
    # -------------------------------------------------------------------------

    def initial_solutions(self, p, d):
        init_solutions = self.init_solutions
        for i in range(p):
            my_solution = np.random.uniform(size=d)>0.5
            init_solutions.append(my_solution)        
        return init_solutions


    # -------------------------------------------------------------------------
    # Enforce pinned items: force fixed_solution[i] == True into the candidate
    # -------------------------------------------------------------------------

    def fixed_elements(self, solution):
        solution = list(solution)
        fixed_solution = self.fixed_solution
        if len(fixed_solution) > 0:
            n = 0
            for n in range(len(fixed_solution)):
                if fixed_solution[n] == True:
                    solution[n] = True
        return solution


    # -------------------------------------------------------------------------
    # Random index guard: protect pinned indices when random_control is True
    # - If random_control == True and target index is pinned -> return False (block)
    # - Otherwise -> return True (allow)
    # -------------------------------------------------------------------------

    def random_index_control(self, target_index):
        random_control = copy.deepcopy(self.random_control)
        if random_control == True:
            fixed_solution = self.fixed_solution
            if len(fixed_solution) > 0:
                if fixed_solution[target_index] == True:
                    random_control = False
        else:
            random_control = True
        return random_control
    
    
    # -------------------------------------------------------------------------
    # Scale rows by per-item statistic
    # - If avg_sqrt_std_control: scale by mean / sqrt(std); else by mean.
    # - Emphasizes stable, high-mean rows before downstream sorting/selection.
    # -------------------------------------------------------------------------

    def convert_data(self, zero_one_matrix):
        d = self.dimension
        for i in range(d):
            avg = statistics.mean(zero_one_matrix[i][:])
            std = statistics.stdev(zero_one_matrix[i][:])
            sqrt_std = math.sqrt(std)
            if self.avg_sqrt_std_control == True:
                item_parameter = avg / sqrt_std
            else:
                item_parameter = avg
            zero_one_matrix[i] = np.dot(zero_one_matrix[i], item_parameter)
        return zero_one_matrix


    # -------------------------------------------------------------------------
    # Scale rows and emit sort parameters
    # - Always compute per-item score = mean / sqrt(std).
    # - Return (scaled matrix, per-item scores) for sorting.
    # -------------------------------------------------------------------------

    def convert_data_and_get_sort_parameters(self, zero_one_matrix):
        d = self.dimension
        sort_parameters = []
        for i in range(d):
            avg = statistics.mean(zero_one_matrix[i][:])
            std = statistics.stdev(zero_one_matrix[i][:])
            sqrt_std = math.sqrt(std)
            item_parameter = avg / sqrt_std 
            sort_parameters.append(item_parameter)
            zero_one_matrix[i] = np.dot(zero_one_matrix[i], item_parameter)
        return zero_one_matrix, sort_parameters


    # -------------------------------------------------------------------------
    # Sort rows by provided importance scores (descending)
    # - Produce sorting_data_index_list via greedy argmax.
    # - Reorder profits_data (rows) and weight_vector accordingly.
    # - Return sorted views + index list for reverse mapping.
    # -------------------------------------------------------------------------
         
    def sort_data (self, profits_data, weight_vector, sort_parameters):
        d = self.dimension
        sorting_data_index_list = [] 
        for item in range(d):
            sort_index = np.argmax(sort_parameters)
            sorting_data_index_list.append(sort_index)
            sort_parameters[sort_index] = -1
        sorting_profits_data = []
        sorting_weights_data = []
        for item in sorting_data_index_list:
            sorting_profits_data.append(profits_data[item])
            sorting_weights_data.append(weight_vector[item])
        return sorting_profits_data, sorting_weights_data, sorting_data_index_list
 

    # -------------------------------------------------------------------------
    # Augmentation (with fixed seeds): ROS -> SMOTE -> reorder
    # - Build df_class from per-sample (>0) counts, then balance via ROS+SMOTE.
    # - If relationship_control is False: scale+score then sort with pinned boost.
    # - Else: relationship-aware projection around pinned items.
    # - Return fixed_solution, profits, weights, and mapping index list.
    # -------------------------------------------------------------------------
       
    def augmented_data(self):
        df_weights = pd.DataFrame(self.weights)
        df_weights.columns = ["weights"]
        df_profits = pd.DataFrame(self.profits)
        df_profits = df_profits.T
        df_class = df_profits[df_profits>0].count(axis = 1)

        ros = RandomOverSampler(random_state=42)
        df_profits, df_class = ros.fit_resample(df_profits, df_class)
        smote = SMOTE()
        df_profits, df_class = smote.fit_resample(df_profits, df_class)
        
        df_profits = df_profits.T
        profits = df_profits.values.tolist()
        weights = df_weights["weights"].values.tolist()
        
        if self.relationship_control == False:
            converted_profits, sort_list = self.convert_data_and_get_sort_parameters(profits)
            fixed_solution, profits, weights, sorting_augmented_index_list_l1 = self.sort_profits_and_solutions(self.dimension, self.fixed_solution, sort_list, converted_profits, weights)
        else:
            self.profits = profits
            self.weights = weights
            fixed_solution, profits, weights, sorting_augmented_index_list_l1 = self.solution_space_of_selected_items(self.dimension, self.fixed_solution, profits, weights)
        return fixed_solution, profits, weights, sorting_augmented_index_list_l1


    # -------------------------------------------------------------------------
    # Augmentation (no fixed seeds): ROS -> SMOTE -> global order
    # - Balance via ROS+SMOTE; compute scores (mean/sqrt(std)) and derive index order.
    # - Returns (raw profits, weights, index list); sorting applied downstream
    #   using the emitted index list.
    # -------------------------------------------------------------------------

    def augmented_data_no_constant(self):
        df_weights = pd.DataFrame(self.weights)
        df_weights.columns = ["weights"]
        df_profits = pd.DataFrame(self.profits)
        df_profits = df_profits.T
        df_class = df_profits[df_profits>0].count(axis = 1)
        
        ros = RandomOverSampler(random_state=42)
        df_profits, df_class = ros.fit_resample(df_profits, df_class)
        smote = SMOTE()
        df_profits, df_class = smote.fit_resample(df_profits, df_class)

        df_profits = df_profits.T
        profits = df_profits.values.tolist()
        weights = df_weights["weights"].values.tolist()
        
        converted_profits, sort_parameters_list = self.convert_data_and_get_sort_parameters(profits)
        sorting_profits, sorting_weights, sorting_augmented_index_list_l1 = self.sort_data(converted_profits, weights, sort_parameters_list)
        return profits, weights, sorting_augmented_index_list_l1

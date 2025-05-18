#!/bin/bash

EXECUTABLE="./15puzzle"
NUM_RUNS=10

CSV_H2="heuristic2_results.csv"
CSV_H3="heuristic3_results.csv"

echo "visited_states,solution_length" > "$CSV_H2"
echo "visited_states,solution_length" > "$CSV_H3"

echo "Running Heuristic 2 (Manhattan Distance) for $NUM_RUNS iterations..."
for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE -2 >> "$CSV_H2"
done

echo "Running Heuristic 3 (Manhattan Distance) for $NUM_RUNS iterations..."
for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE -3 >> "$CSV_H3"
done

echo "gg wp"

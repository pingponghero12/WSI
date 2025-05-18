#!/bin/bash

EXECUTABLE="./build/solver"

# 8
NUM_RUNS=1000

CSV_8_H1="8h1.csv"
CSV_8_H2="8h2.csv"
CSV_8_H3="8h3.csv"

echo "visited_states,solution_length" > "$CSV_8_H1"
echo "visited_states,solution_length" > "$CSV_8_H2"
echo "visited_states,solution_length" > "$CSV_8_H3"

for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE 3 -1 >> "$CSV_8_H1"
done

for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE 3 -2 >> "$CSV_8_H2"
done

for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE 3 -3 >> "$CSV_8_H3"
done

# 15
# It is worthless to try, I will get one problem where it takes ages and more importantly all of my ram and when OS tries to load everything to swp then it's jover.
CSV_15_H3="15h3.csv"
NUM_RUNS=1000

echo "visited_states,solution_length" > "$CSV_15_H3"

for (( i=0; i<NUM_RUNS; i++ )); do
    $EXECUTABLE 4 -3 >> "$CSV_15_H3"
done

echo "gg wp"

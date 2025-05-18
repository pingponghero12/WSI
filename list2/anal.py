import pandas as pd

def calculate_and_print_averages(csv_filename, heuristic_name):
    try:
        df = pd.read_csv(csv_filename)
        
        avg_visited_states = df['visited_states'].mean()
        
        successful_runs = df[df['solution_length'] != -1]
        if not successful_runs.empty:
            avg_solution_length = successful_runs['solution_length'].mean()
        else:
            avg_solution_length = float('nan') 

        print(f"{heuristic_name}:")
        print(f"  Average Visited States: {avg_visited_states:.2f}")
        print(f"  Average Solution Length (successful runs): {avg_solution_length:.2f}")
        
    except FileNotFoundError:
        print(f"Error: {csv_filename} not found.")
    except Exception as e:
        print(f"Error processing {csv_filename}: {e}")

if __name__ == "__main__":
    calculate_and_print_averages("8h1.csv", "Heuristic 1 (Misplaced Tiles)")
    print("-" * 30)
    calculate_and_print_averages("8h2.csv", "Heuristic 2 (Manhattan Distance)")
    calculate_and_print_averages("8h3.csv", "Heuristic 2 (Linear Conflicts)")

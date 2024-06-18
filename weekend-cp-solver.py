import argparse
import csv
import math
import os
from ortools.sat.python import cp_model
from tabulate import tabulate

# Constants for scoring weights
PREFERRED_WEIGHT = 10  # Higher score for preferred slots
AVAILABLE_WEIGHT = 1   # Lower score for just available slots
DUMMY_EMAIL = '?@datastax.com'
DEBUG = False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve weekend shift scheduling from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing sign-up data.')
    return parser.parse_args()

def validate_file_path(file_path):
    """Check if the file exists and is a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist or is not a file.")

def read_csv_header_and_rows(csv_file):
    """Read and return the header of the CSV file."""
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        return header, list(reader)

def count_shift_types(header):
    """Count weekend and holiday shifts based on the header of the CSV."""
    weekend_count = 0
    holiday_count = 0
    last_weekend_index = -1
    first_holiday_index = len(header)  # Initialize to an index after the last possible index

    for i, column_name in enumerate(header):
        if 'Weekend' in column_name:
            weekend_count += 1
            last_weekend_index = i
        elif 'Holiday' in column_name:
            holiday_count += 1
            if i < first_holiday_index:
                first_holiday_index = i

    # Verify that all 'Holiday' columns are after all 'Weekend' columns
    if holiday_count > 0 and first_holiday_index < last_weekend_index:
        raise ValueError("Error: There are 'Holiday' columns before 'Weekend' columns.")

    print(f"Weekend shifts count: {weekend_count}")
    print(f"Holiday shifts count: {holiday_count}")
    return weekend_count, holiday_count

def collect_availability(rows, should_add_dummy_entries):
    """Collect the availability and preferences from the rows."""
    availables = dict()
    preferences = dict()
    for row in rows:
        email = row[1]
        for column in range(2, len(header)):
            match row[column]:
                case 'Meh':
                    availables.setdefault(email, []).append(column - 2)
                case 'Preferred':
                    preferences.setdefault(email, []).append(column - 2)
                    availables.setdefault(email, []).append(column - 2)
                case _:
                    pass
    if should_add_dummy_entries:
        for column in range(2, len(header)):
            availables.setdefault(DUMMY_EMAIL, []).append(column - 2)

    return availables, preferences

def setup_model(all_shifts, preferences, availables, max_shifts_per_engineer, weekend_count, holiday_count, max_dummy_shifts):
    """Setup the constraint programming model."""
    model = cp_model.CpModel()
    slots = {email: [model.NewBoolVar(f'{email}_slot_{slot}') for slot in range(all_shifts)] for email in availables}

    # Calculate scores for each slot
    slot_scores = calculate_slot_scores(all_shifts, preferences, availables)

    # Add the objective function to maximize the total score
    total_score = sum(slot_scores[email][slot] * slots[email][slot] for email in slots for slot in range(all_shifts))
    model.Maximize(total_score)

    # Add constraints
    add_constraints(model, slots, max_shifts_per_engineer, weekend_count, holiday_count, max_dummy_shifts)

    return model, slots

def calculate_slot_scores(all_shifts, preferences, availables):
    """Calculate scores for each slot based on preferences and availability."""
    slot_scores = {email: [-100] * all_shifts for email in availables}
    for email, slots_list in preferences.items():
        for slot in slots_list:
            slot_scores[email][slot] = PREFERRED_WEIGHT
    for email, slots_list in availables.items():
        for slot in slots_list:
            # Only update the score if it's not already marked as preferred
            if slot_scores[email][slot] != PREFERRED_WEIGHT:
                slot_scores[email][slot] = AVAILABLE_WEIGHT
    return slot_scores

def add_constraints(model, slots, max_shifts_per_engineer, weekend_count, holiday_count, max_dummy_shifts):
    """Add constraints to the model."""
    # Prevent more than one engineer being assigned to a slot
    for slot in range(weekend_count):
        model.Add(sum(slots[email][slot] for email in slots) == 1)
    
    # Two engineers should be assigned to holiday slots:
    for slot in range(weekend_count, weekend_count + holiday_count):
        model.Add(sum(slots[email][slot] for email in slots) == 2)

    # Prevent an engineer being assigned more than the maximum shifts allowed
    for email in slots:
        if email != DUMMY_EMAIL:
            model.Add(sum(slots[email]) == max_shifts_per_engineer)
        else:
            model.Add(sum(slots[email]) == max_dummy_shifts)

    for email in slots:
        if email != DUMMY_EMAIL:
            # Prevent assignment of more than one shift in a given weekend, as well as being assigned back to back weekends
            for slot in range(0, weekend_count // 2, 2):  # Iterate over first half of available weekend slots (every other shift)
                if slot + weekend_count // 2 + 3 < weekend_count:
                    model.Add(
                        slots[email][slot]
                            + slots[email][slot + 1]
                            + slots[email][slot + 2]
                            + slots[email][slot + 3]
                            + slots[email][slot + weekend_count // 2]
                            + slots[email][slot + weekend_count // 2 + 1]
                            + slots[email][slot + weekend_count // 2 + 2]
                            + slots[email][slot + weekend_count // 2 + 3]
                        <= 1
                    )
            # Prevent assignment of more than one shift on a given holiday
            for slot in range(weekend_count, weekend_count + holiday_count , 2):
                model.Add(
                    slots[email][slot]
                        + slots[email][slot + 1]
                        + slots[email][slot + holiday_count]
                        + slots[email][slot + holiday_count + 1]
                    <= 1
                )
    
def solve_model(model):
    """Solve the model and return the solver status and solution."""
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status, solver

def print_shift_assignments(assigned_slots, weekend_count, holiday_count, csv_columns):
    """Print the shift assignments in a table format."""
    transformed_data = {}
    assignment_counts = {}
    for engineer, entries in assigned_slots.items():
        for slot, choice in entries:
            transformed_data.setdefault(slot, []).append({'engineer': engineer, 'choice': choice})
            assignment_counts[engineer] = assignment_counts.get(engineer, 0) + 1
    if DEBUG:
        # Print the number of assignments for each engineer
        print("\nAssignments per engineer:")
        for engineer, count in assignment_counts.items():
            print(f"{engineer}: {count}")
    if weekend_count > 0:
        headers = ['Dates', 'Saturday Early', 'Sunday Early', 'Saturday Late', 'Sunday Late']
        rows = []
        for i in range(0, weekend_count // 2, 2):
            row = []
            row.append(csv_columns[i].split('[')[1].split(' ')[0] + ' -> ' + csv_columns[i + 1].split('[')[1].split(' ')[0])
            for slot in [i, i + 1, i + weekend_count // 2, i + weekend_count // 2 + 1]:
                if slot in transformed_data:
                    row.append(f'{transformed_data[slot][0]["engineer"]} ({transformed_data[slot][0]["choice"]})')
                else:
                    row.append('')
            rows.append(row)
        print('\n' + tabulate(rows, headers=headers))
    if holiday_count > 0:
        headers = ['Dates','Holiday Early 1', 'Holiday Early 2', 'Holiday Late 1', 'Holiday Late 2']
        rows = []
        for i in range(weekend_count, weekend_count + holiday_count // 2):
            row = []
            row.append(csv_columns[i].split('[')[1].split(' ')[0])
            for slot in [i, i + holiday_count // 2]:
                if slot in transformed_data and len(transformed_data[slot]) >= 1:
                    row.append(f'{transformed_data[slot][0]["engineer"]} ({transformed_data[slot][0]["choice"]})')
                    if len(transformed_data[slot]) == 2:
                        row.append(f'{transformed_data[slot][1]["engineer"]} ({transformed_data[slot][1]["choice"]})')
                    else:
                        row.append('')
                else:
                    row.append('')
                    row.append('')
            rows.append(row)
        print('\n' + tabulate(rows, headers=headers))

try:
    args = parse_arguments()
    validate_file_path(args.csv_file)
    header, rows = read_csv_header_and_rows(args.csv_file)
    weekend_count, holiday_count = count_shift_types(header)
    all_shifts = weekend_count + holiday_count * 2
    max_shifts_per_engineer = math.floor(all_shifts / len(rows))
    max_dummy_shifts = all_shifts % len(rows)
    availables, preferences = collect_availability(rows, max_dummy_shifts != 0)

    model, slots = setup_model(all_shifts, preferences, availables, max_shifts_per_engineer, weekend_count, holiday_count, max_dummy_shifts)
    status, solver = solve_model(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Truncate and store emails with their assigned slots and preference status
        truncated_emails_assigned_slots = {}
        for email, slots_list in availables.items():
            if DEBUG:
                print(email, slots_list)
                for slot in slots[email]:
                    if solver.Value(slot):
                        print(f"{email} assigned to slot {slot}")
            assigned_slots = [(slot, 'P' if slot in preferences.get(email, []) else 'A') for slot in slots_list if solver.Value(slots[email][slot])]
            # Truncate email to first name initial and last name initial
            name_parts = email.split('@')[0].split('.')
            truncated_email = f'{name_parts[0]}.{name_parts[-1][0]}'  # Corrected to use initials
            truncated_emails_assigned_slots[truncated_email] = assigned_slots
        print_shift_assignments(truncated_emails_assigned_slots, weekend_count, holiday_count, header[2:])
    else:
        print('No solution found')
except Exception as e:
    print(f"An error occurred while processing the file: {e}")
    exit(1)
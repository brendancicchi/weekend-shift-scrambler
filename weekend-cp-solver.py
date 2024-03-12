import argparse
import csv
import math
import os
from ortools.sat.python import cp_model

# Constants for scoring weights
PREFERRED_WEIGHT = 10  # Higher score for preferred slots
AVAILABLE_WEIGHT = 1   # Lower score for just available slots

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Solve weekend shift scheduling from a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file containing sign-up data.')
    return parser.parse_args()

def validate_file_path(file_path):
    """Check if the file exists and is a file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Error: The file '{file_path}' does not exist or is not a file.")

def read_csv_header(csv_file):
    """Read and return the header of the CSV file."""
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        return next(reader)

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

def setup_model(all_shifts, preferences, availables, max_shifts_per_engineer, weekend_count):
    """Setup the constraint programming model."""
    model = cp_model.CpModel()
    slots = {email: [model.NewBoolVar(f'{email}_slot_{slot}') for slot in range(all_shifts)] for email in availables}

    # Calculate scores for each slot
    slot_scores = calculate_slot_scores(all_shifts, preferences, availables)

    # Add the objective function to maximize the total score
    total_score = sum(slot_scores[email][slot] * slots[email][slot] for email in slots for slot in range(all_shifts))
    model.Maximize(total_score)

    # Add constraints
    add_constraints(model, slots, max_shifts_per_engineer, weekend_count)

    return model, slots

def calculate_slot_scores(all_shifts, preferences, availables):
    """Calculate scores for each slot based on preferences and availability."""
    slot_scores = {email: [0] * all_shifts for email in availables}
    for email, slots_list in preferences.items():
        for slot in slots_list:
            slot_scores[email][slot] = PREFERRED_WEIGHT
    for email, slots_list in availables.items():
        for slot in slots_list:
            # Only update the score if it's not already marked as preferred
            if slot_scores[email][slot] != PREFERRED_WEIGHT:
                slot_scores[email][slot] = AVAILABLE_WEIGHT
    return slot_scores

def add_constraints(model, slots, max_shifts_per_engineer, weekend_count):
    """Add constraints to the model."""
    # Prevent more than one engineer being assigned to a slot
    for slot in range(all_shifts):
        model.Add(sum(slots[email][slot] for email in slots) == 1)

    # Prevent an engineer being assigned more than the maximum shifts allowed
    for email in slots:
        model.Add(sum(slots[email]) <= max_shifts_per_engineer)

    # Prevent assignment of more than one shift in a given weekend, as well as being assigned back to back weekends
    for email in slots:
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

def solve_model(model):
    """Solve the model and return the solver status and solution."""
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    return status, solver

try:
    args = parse_arguments()
    validate_file_path(args.csv_file)
    header = read_csv_header(args.csv_file)
    weekend_count, holiday_count = count_shift_types(header)
    all_shifts = weekend_count + holiday_count

    with open(args.csv_file, 'r') as file:
        availables = dict()
        preferences = dict()
        unavailables = dict()

        reader = csv.reader(file)
        # Skip the second row and header
        next(reader)
        next(reader)

        # Count the number of engineers
        engineers_list = list(reader)

        # Max shifts per engineer 
        max_shifts_per_engineer = math.ceil(all_shifts / len(engineers_list))

        # Shift numbers assigned are 0 -> (num_columns - 2)
        for row in engineers_list:
            email = row[1]
            for column in range(2, len(header)):
                match row[column]:
                    case 'Meh':
                        availables.setdefault(email, []).append(column - 2)
                    case 'Preferred':
                        preferences.setdefault(email, []).append(column - 2)
                        availables.setdefault(email, []).append(column - 2)
                    case 'Unavailable':
                        unavailables.setdefault(email, []).append(column - 2)

    model, slots = setup_model(all_shifts, preferences, availables, max_shifts_per_engineer, weekend_count)
    status, solver = solve_model(model)

    def print_assignment_table(truncated_emails_assigned_slots, weekend_count, holiday_count):
        # Print header for the table
        print(f'\n{"Saturday Early"} | {"Sunday Early"} | {"Saturday Late"} | {"Sunday Late"}')
        print('-' * 64)  # Print a separator line
        # Sort and print emails by the specified order in a table format
        for slot_group_start in range(0, weekend_count // 2, 2):
            slot_group_end = slot_group_start + 1
            mirror_slot_group_start = slot_group_start + weekend_count // 2
            mirror_slot_group_end = mirror_slot_group_start + 1

            # Collect emails for the current group with preference status
            emails_for_group = {}
            for email, slots in truncated_emails_assigned_slots.items():
                for slot, pref_status in slots:
                    if slot_group_start == slot:
                        emails_for_group.setdefault("Saturday Early", []).append(f'{email}({pref_status})')
                    if slot_group_end == slot:
                        emails_for_group.setdefault("Sunday Early", []).append(f'{email}({pref_status})')
                    if mirror_slot_group_start == slot:
                        emails_for_group.setdefault("Saturday Late", []).append(f'{email}({pref_status})')
                    if mirror_slot_group_end == slot:
                        emails_for_group.setdefault("Sunday Late", []).append(f'{email}({pref_status})')

            # Sort the emails for the current group
            for day in emails_for_group:
                emails_for_group[day].sort()

            # Print the current row of the table
            saturday_early_emails = " ".join(emails_for_group.get("Saturday Early", []))
            sunday_early_emails = " ".join(emails_for_group.get("Sunday Early", []))
            saturday_late_emails = " ".join(emails_for_group.get("Saturday Late", []))
            sunday_late_emails = " ".join(emails_for_group.get("Sunday Late", []))
            print(f'{saturday_early_emails:<14} | {sunday_early_emails:<14} | {saturday_late_emails:<14} | {sunday_late_emails:<14}')
        
        if all_shifts > weekend_count:
            # Print header for the table
            print(f'\n{"Holiday Early"} | {"Holiday Late"}')
            print('-' * 64)  # Print a separator line
            for slot_group_start in range(weekend_count // 2, (weekend_count + holiday_count) // 2, 2):
                slot_group_end = slot_group_start + 1
                mirror_slot_group_start = slot_group_start + holiday_count // 2
                mirror_slot_group_end = mirror_slot_group_start + 1

                # Collect emails for the current group with preference status
                emails_for_group = {}
                for email, slots in truncated_emails_assigned_slots.items():
                    for slot, pref_status in slots:
                        if slot_group_start == slot or slot_group_end == slot:
                            emails_for_group.setdefault("Holiday Early", []).append(f'{email}({pref_status})')
                        if mirror_slot_group_start == slot or mirror_slot_group_end == slot:
                            emails_for_group.setdefault("Holiday Late", []).append(f'{email}({pref_status})')

                # Sort the emails for the current group
                for day in emails_for_group:
                    emails_for_group[day].sort()

                # Print the current row of the table
                holiday_early_emails = ", ".join(emails_for_group.get("Holiday Early", []))
                holiday_late_emails = ", ".join(emails_for_group.get("Holiday Late", []))
                print(f'{holiday_early_emails:<28} | {holiday_late_emails:<28}')
        



    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # Truncate and store emails with their assigned slots and preference status
        truncated_emails_assigned_slots = {}
        for email, slots_list in availables.items():
            assigned_slots = [(slot, 'P' if slot in preferences.get(email, []) else 'A') for slot in slots_list if solver.Value(slots[email][slot])]
            # Truncate email to first name initial and last name initial
            name_parts = email.split('@')[0].split('.')
            truncated_email = f'{name_parts[0]}.{name_parts[-1][0]}'  # Corrected to use initials
            truncated_emails_assigned_slots[truncated_email] = assigned_slots

        print_assignment_table(truncated_emails_assigned_slots, weekend_count, holiday_count)
    else:
        print('No solution found')
except Exception as e:
    print(f"An error occurred while processing the file: {e}")
    exit(1)
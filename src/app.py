import os
import argparse

# Create an argument parser
parser = argparse.ArgumentParser(description="A simple app to demonstrate Docker and env vars.")
parser.add_argument("--message", type=str, default="Hello MLOps!", help="Message to display.")

args = parser.parse_args()

# Get an environment variable, with a default fallback
student_name = os.getenv("STUDENT_NAME", "Future MLOps Engineer")

print("========================================")
print(f"Message: {args.message}")
print(f"Brought to you by: {student_name}")
print("========================================")

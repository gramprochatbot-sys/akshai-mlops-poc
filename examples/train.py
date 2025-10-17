"""
smoke test
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Run a fast sanity check training")
args = parser.parse_args()

if args.smoke_test:
    # Run a very short training loop
    N_EPOCHS = 1
    SUBSET_SIZE = 50
else:
    # Normal full training
    N_EPOCHS = 100
    SUBSET_SIZE = None

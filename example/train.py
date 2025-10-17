import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--smoke-test", action="store_true", help="Run a fast sanity check training")
args = parser.parse_args()

if args.smoke_test:
    # Run a very short training loop
    n_epochs = 1
    subset_size = 50
else:
    # Normal full training
    n_epochs = 100
    subset_size = None

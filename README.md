# Conditional RV-GOMEA

## Building

Run `make` in this directory to build the project. The Armadillo library needs to be installed on the system.

## Running

To start a single run of the algorithm, run the command below. This will give you an overview of options with which you can control the algorithm.

```shell
python rvgomea/cmd/run.py --help
```

## Testing

To run a quick test of different linkage models, try the following command:

```shell
make && run_scripts/test_all_models
```


Manual
-------------------------

Code for manually running benchmarks. More boilerplate, but also more flexibility compared to the benchmark
interface.

We provide an example that replicates a commonly used benchmark that includes 12 Out-of-Distribution detectors,
each tested against 9 OOD datasets.
We subsequently calculate the average performance of each detector
across all datasets and sort the outcomes based on their
Area Under Receiver Operating Characteristic (AUROC) score in ascending order.

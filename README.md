# BABOONS: Black-Box Optimization of Data Summaries in Natural Language

BABOONS optimizes summaries of data sets in natural language, using language models to evaluate data summaries. For instance, users can submit natural language instructions, describing the type of data summary they are seeking. The system then uses language models to compare alternative summaries for the same data set, selecting the summary that most closely matches user instructions.

The implementation can be used in two modes. BABOONS features an interactive demo interface, allowing users to submit natural language instructions live in a GUI. This interface can be found in the `src/cp/interface/gui.py` file and started via `streamlit run src/cp/interface/gui.py` (note that streamlit needs to be installed first). A video demonstrating this interface is available [here](https://www.youtube.com/embed/ssGwZcUkMKA). 

The experiments presented in the associated VLDB paper can be reproduced using the files in the `src/cp/eval` folder. Execute `run_single.py` to run experiments that summarize different items separately. Before doing so, make sure to uncomment all relevant baselines in the main function (starting from Line 232). The data to summarize must be stored in a Postgres database. The script takes several input parameters:
- The name of the Postgres database containing data to summarize.
- The user name of the Postgres database (it is assumed that no password is required).
- The name of the output file, containing generated data summaries.
- The logging level, determining how much output is generated during processing.
- The number of RL samples used per test case (e.g., 200 is a reasonable choice in most cases).

The script will iterate over different summary parameters, varying the number of facts per summary and the number of predicates per fact. It will run all uncommented baselines on each of the test cases. The test cases are described in the file `bench.py` via the data structures (`scenarios`) defined at the beginning of the file. For each scenario, the file defines columns used as dimensions and for aggregates in the data summary. Also, it defines text templates used to generate data summaries. Finally, for each scenario, it defines a set of items that are summarized as part of the evaluation.

Use `run_batch.py` to run experiments that treat batches of items as a whole, thereby gaining efficiency. This script takes the following input parameters:
- The path to the JSON input, describing the batch of items to summarize. JSON files for the paper scenarios are given in the `bench` folder within the root directory. The current implementation only supports items defined via SQL equality predicates (i.e., it does not cover all four scenarios). The paper experiments are based on the third scenario, the associated description is in `batch3.json`.
- The name of the Postgres database containing data to summarize.
- The user name to access the Postgres database (no password can be specified).
- A path to the output file, containing generated descriptions.
- The log level, determining how much output to generate during processing.

## How to Cite
```
@article{Trummer2022f,
author = {Trummer, Immanuel},
journal = {PVLDB},
number = {11},
pages = {2980 -- 2993},
title = {{BABOONS: Black-box optimization of data summaries in natural language}},
volume = {15},
year = {2022}
}
```

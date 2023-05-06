# NaturalMiner

NaturalMiner mines data for patterns described in natural language. E.g., let's say you have a table with data about laptop models. For a specific laptop, you can mine for

```
arguments for buying the laptop
```

NaturalMiner automatically maps your pattern to relevant SQL queries, executes them efficiently, and retrieves the most relevant facts, given your pattern. To compare mined facts to your input pattern, it uses large language models such as BERT or GPT.

# Quickstart

Try out NaturalMiner using the Notebook [here](https://colab.research.google.com/drive/1EYbdlVgzOkf0b6PTjzntfRi_aEQaups_?usp=sharing). The notebook loads a sample database and allows you to mine for different patterns.

# Local Setup

**The following commands have been tested on Ubuntu 20 and should work on other Linux distributions as well as Mac OS X.**

Install NaturalMiner via pip:
```
pip install naturalminer
```

If you want to use NaturalMiner via a graphical user interface in your Web browser, use the following command instead:
```
pip install naturalminer[gui]
```

NaturalMiner mines data stored in a Postgres database. If you have not installed Postgres yet, you can install and start it on Ubuntu using the following commands:
```
!sudo apt-get -y -qq update
!sudo apt-get -y -qq install postgresql
!sudo service postgresql start
```

# Using NaturalMiner

You find an invocation example under `src/nminer/interface/example.py`. It uses the `mine` function to mine relevant facts. Before using NaturalMiner, make sure to load the relevant data into a Postgres database. NaturalMiner mines data in single tables. You find the SQL script creating the example data [here](https://drive.google.com/file/d/1pB6c8XnWF65vKUlDTiFVR5oeUPx9X0pN/view?usp=sharing). Create an example database and run the script (from the directory containing it) using the following commands:

```
createdb picker
psql -f laptops.sql picker
```

Next, we discuss the parameters of the mining function. You may have to update the default values set in `example.py`, depending on your local setup.

## Configuring Data Access

- `db_name`: the name of the Postgres database.
- `db_user`: the name of a Postgres user with access to the database.
- `db_pwd`: the Postgres password of the database user.
- `table`: name of the table to mine in the Postgres database.

## Configuring Data Semantics

NaturalMiner assesses relevance based on a natural language description of mined facts. You need to provide text templates allowing NaturalMiner to express mined facts in natural language.

- `preamble`: prefix text used at the start of each mined fact. E.g., if mining facts that compare laptops, this could be `Among all laptops`. 
- `dim_cols`: list of table column names to consider for equality predicates.
- `dims_txt`: list of text templates for expressing equality predicates on aforementioned columns. Each text template is a string that contains a placeholder to represent the constant in the equality predicate. E.g., `with <V> graphics card` is a reasonable template for expressing a restriction to laptop models with a specific graphics card (which substitutes the `<V>` placeholder).
- `agg_cols`: list of numerical table columns to consider for aggregation.
- `aggs_txt`: list of text descriptions associated with aggregation columns. E.g., `its discounted price` is a reasonable description for a column containing the laptop price.

## Configuring Mining Goals

- `target`: NaturalMiner mines for facts that compare target data to the entire data set. E.g., the target data could relate to one specific laptop model. The target is a string containing an SQL predicate referencing the input table (e.g., `laptop_name='VivoBook S430'`).
- `nl_pattern`: NaturalMiner retrieves facts matching a pattern described in natural language. E.g., `arguments for buying the laptop` or `why this laptop is bad` are possible patterns. Don't worry whether those patterns map to specific SQL queries - NaturalMiner takes care of that automatically.

## Configuring the Mining Process

Optionally, you can configure the mining process. While the default settings are reasonable, you probably want to try different configurations to get optimal performance.

- `hg_model`: ID of the language model used to compare facts to the input pattern. NaturalMiner supports models for zero-shot classificatio, available on the [Huggingface model hub] (https://huggingface.co/models). E.g., `facebook/bart-large-mnli' is the default.
- `nr_facts`: NaturalMiner searches single facts of combinations of facts. This parameter chooses the number of facts.
- `nr_preds`: Facts relate to SQL queries that can use up to this many predicates. E.g., a fact stating that the average price of a laptop is 20% lower, compared to other laptops of the same brand and screen size uses two predicates (restrictions on brand and screen size).
- `degree`: NaturalMiner represents the search space as a graph where nodes represent fact combinations, edges connect similar facts. The degree determines the number of neighbor nodes in this graph. The default setting of five should work well for most scenarios.
- `nr_iterations`: NaturalMiner iteratively evaluates fact combinations for a given number of iterations. The default setting of 200 works well for relatively small data sets. If the number of rows or columns is large, you may want to increase this parameter for optimal quality.

# Using the GUI

NaturalMiner can also be used over a GUI in the Web browser. To use the GUI, you need to install NaturalMiner with the GUI option (see above). Then, from the root directory, start the GUI using the following command:
```
streamlit run src/nminer/interface/gui.py
```
The terminal should now show a URL, allowing you to access the GUI using a Web browser. The interface enables you to change most of the aforementioned configuration options directly in the GUI.

# Reproducing Paper Experiments

The experiments presented in the associated VLDB paper can be reproduced using the files in the `src/nminer/eval` folder. Execute `run_single.py` to run experiments that summarize different items separately. Before doing so, make sure to uncomment all relevant baselines in the main function (starting from Line 232). The data to summarize must be stored in a Postgres database, you find the corresponding data [here](https://drive.google.com/file/d/131r8WJexU1JsmIL4Gyx9EFIZXdOTbeX7/view?usp=sharing). The script takes several input parameters:
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

Please cite the following publication if refering to this code (the system was renamed from BABOONS to NaturalMiner):

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

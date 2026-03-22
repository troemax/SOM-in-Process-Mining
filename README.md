# SOM in Process Mining

This tool creates and visualizes self-organizing maps (SOMs) trained on event logs (.xes files).

## Setup

This tool was programmed using Python version 3.13.7. The list of packages needed can be found in the environment.yml file.

## Running

To run the tool, execute the main.py file. A website will open automatically in your standard web browser. There, you can load any event log which was placed in the folder "inputs".

## WARNING

The tool should only be used by people you trust, because it allows to execute arbitrary code. The feature to create custom aggregations allows the user to input Python code that gets executed without any checks. This could pose a severe security risk for your system.

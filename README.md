# Model Selection for Forecasting

Forecasting daily aggregate and hourly Direct Normal(DNI) using various methods.

Course project for ASL 760 - Renewable Energy Meteorology

Data is expected to be from one site, and in csv format

### Running
```bash
python Analyse.py --data_dir <Data Folder>
```

or
    
```bash
python Analyse.py --data <csv file with proper headers and timestamp>
```

### For options
```bash
python Analyse.py -h
```



### Input variations tried (graphs in the 'Graphs' folder) -

* Predicting daily and hourly aggregate(summed) DNI based on past 5 years' DNI on the given date
* Predicting daily aggregate DNI based on past 30 days' DNI
* Predicting daily and hourly aggregate DNI based on past 5 years' DNI & GHI on the given date
* Predicting daily aggregate DNI based on past 30 days' DNI & GHI


### Data source
Cooperative Networks For Renewable Resource Measurements Solar Energy Resource Data - http://rredc.nrel.gov/solar/new_data/confrrm/

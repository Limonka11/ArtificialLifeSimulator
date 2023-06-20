# ArtificialLifeSimulator
In order to simulate the evolution and intellectual advancements
of species we combine state of the art reinforcment learning algorithm PPO
and evolutionary algorithms.

##  2. Getting Started
To get started, you will need to install the requirements and 
pull the content of this repository.

###  2.1. Prerequisites
To install the requirements, simply run the following:  
```pip install -r requirements.txt```

###  2.2. Usage
There are many tunable parameters in this project, therefore, we advise
you to start with the pre-configures values and then tune them as appropriate.

#### Training
To train a model, simply run the train.py file. You could configure values
from the command line. To see more about the configurable values you can run:
```python .\train.py -h```

#### Testing
To test a model, simply run the train.py file again. However, this time
you need to specify the file containing the model that you are trying to test:
```python .\test.py --test-model '...'```

#### Tuning
To tune a model, simply run the tune.py. There are again many values
to configure. To see more about the configurable values you can run:
```python .\tune.py -h```

#### Video Examples
- Video of wolves performing collaborative hunting!
  
[![Video of wolves performing collaborative hunting](http://i3.ytimg.com/vi/Lsw4edRWRMw/hqdefault.jpg)](https://www.youtube.com/watch?v=Lsw4edRWRMw)

- Video of wolves waiting for agents to drink near the lake and fighting amongst themselves!
  
[![Video of wolves performing collaborative hunting](http://i3.ytimg.com/vi/Lsw4edRWRMw/hqdefault.jpg)](https://www.youtube.com/shorts/FAXsG4GofEQ)

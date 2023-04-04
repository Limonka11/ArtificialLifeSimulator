from setuptools import setup

# This environment depents on the Gym library
# and its source code will be in the gym_example subdirectory
setup(name="gym_example",
      version="1.0.0",
      install_requires=["gymnasium", "pygame"]
)
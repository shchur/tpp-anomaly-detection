from setuptools import find_packages, setup

setup(name="anomaly_tpp",
      version="0.1.0",
      description="Detecting anomalous event sequences with temporal point processes",
      packages=find_packages("."),
      zip_safe=False)

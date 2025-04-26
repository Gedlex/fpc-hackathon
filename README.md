# FPC-Hackathon
This repository contains the Python code for the hackathon at the "[Future PhD in Control Workshop](https://www.tu-ilmenau.de/workshop-obc)".

## Prerequisites

- Python 3.9
- Numpy
- Scipy
- Casadi
- Control
- Matplotlib
- Conda (optional)

## Installation

1. Download and install [Python](https://www.python.org/downloads/)
2. Create a [Conda](https://docs.anaconda.com/miniconda/miniconda-install/) environment to make sure all the necessary packages are installed

    ```bash
        conda env create --file environment.yml
        conda activate fpc-hackathon
    ```

   or install the packages manually. To update the environment execute

    ```bash
        conda activate fpc-hackathon
        conda env update --file environment.yml --prune
    ```

4. Clone this repository or download the code as a ZIP archive and extract it to a folder of your choice.

## Running Jupyter Notebooks

Start a jupyter notebook server by running

```bash
    jupyter notebook 
```

This requires that you have installed all the packages from `environment.yml`.

## License

This project is licensed under the MIT License.

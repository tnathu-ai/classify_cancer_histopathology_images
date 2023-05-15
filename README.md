# Classify Images of Cancer

Assignment 2: Machine Learning Project

## 🤝‍Authors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/tnathu-ai"><br /><sub><b>Tran Ngoc Anh Thu</b></sub></a><br />s3879312</td>
    <td align="center"><a href="https://github.com/phong-s3879069"><br /><sub><b>Tran Duy Phong</b></sub></a><br />s3879069</td>
    </tr>
</table>

# Introduction

The dataset description contains 27x27 RGB images of cells from 99 different patients. The images are histopathology images that show cells at the microscopic level. The goal is to classify these cells based on whether they are cancerous and also to classify them according to their cell type.

# Instructions

1. Clone this repository to your local machine.

2. Ensure that you have the necessary packages installed. These can be found in the `requirements.txt` file. To install them, use the following command in your terminal:

```python
pip install -r requirements.txt
```

3. Navigate to the `jupyter_notebook` directory and open the `notebook.ipynb` file. This Jupyter notebook contains all the code and detailed analysis for this project.

4. If you want to test the models, you can find the trained models in the `jupyter_notebook/models` directory.

5. You can find the evaluation reports for the data in `independent_evaluation` directory.

6. `output_data` folder contains the processed data used for the project. This includes a CSV file of combined data and evaluation reports.

7. For details on the project requirements and grading rubric, refer to the `requirement_rubric` directory.

Please note that this project requires Python 3.6 or later and pip for installing packages. If you encounter any issues, please feel free to open an issue in this repository.

# Repository Structure

Here is an overview of the repository structure:

```
├── Image_classification_data.zip           # Original data used for the project.
├── LICENSE                                 # License for the project.
├── README.md                               # This README file.
├── data_source.pdf                         # Documentation of the data source.
├── independent_evaluation                  # Directory containing pdfs of independent evaluation metrics.
├── jupyter_notebook                        # Jupyter notebook containing the code and analysis.
│   ├── models                              # Trained models from the project.
│   ├── my_dir                              # Directory containing tuning results.
│   ├── notebook.ipynb                      # The main Jupyter notebook with the entire analysis and code.
│   └── tuning                              # Directory containing tuning results.
│   └── notebook.pdf                        # Pdf file of the jupyter notebook
│   └── notebook.py                         # python file of the jupyter notebook.
├── output_data                             # Output data used for the project
├── requirement_rubric                      # Assignment requirement and rubric.
└── requirements.txt                        # Required packages for the project.
```

# Data source

- https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7399414&tag=1

# License

This project is licensed under the terms of the MIT License.

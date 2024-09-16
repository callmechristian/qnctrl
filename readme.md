# Quantum Network Control

This project is based on my thesis at KTH, available at [link TBD](/). Do check it out to better understand what's going on here.

## Table of Contents

- [Quantum Network Control](#quantum-network-control)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
    ```sh
    https://github.com/callmechristian/qnctrl.git
    cd qnctrl
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Running Notebooks

1. Open Visual Studio Code.
2. Open the desired Jupyter Notebook file (e.g., `CTRL_RL_PPO_continous.ipynb`).
3. Run the cells sequentially to execute the code.

### Modules

The simulator comes pre-equipped with a few modules. You can find all of them in `environment\`.
- core: core simulator modules e.g. entangler, measurement, polarisation controller architectures
- control: additional control method redefinitions
- models: pre-defined models with. name indicates specificity. `sinusoidal_control_fixed.py` recommended as a baseline
- random_motion: noise models
- weather: weather noise models

## Project Structure

- **Data Folders**
  - `data\`
  - `output\`

- **Modules Folder**
  - `environment\`

- **Control Notebooks**:
  - `CTRL_DDPG_simple.ipynb`
  - `CTRL_DNN.ipynb`
  - `CTRL_FFT_DNN.ipynb`
  - `CTRL_Inverse.ipynb`
  - `CTRL_RL_DDPG_sin_sigspace.ipynb`
  - `CTRL_RL_DDPG_sin.ipynb`
  - `CTRL_RL_DDPG.ipynb`
  - `CTRL_RL_DQN.ipynb`
  - `CTRL_RL_PPO_continous.ipynb`
  - `CTRL_RL_REINFORCE.ipynb`

- **Prediction Notebooks**:
  - `PREDICTION_RNN_each_angle.ipynb`
  - `PREDICTION_RNN.ipynb`
  - `PREDICTION_SARIMA.ipynb`

- **Analysis Notebooks**:
  - `ANALYSIS_weatherdata.ipynb`
  - `preview_output.ipynb`

- **Sample Notebooks**
  - `SAMPLE_noise.ipynb`
  - `SAMPLE_weatherdata_interp.ipynb`

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
> [!CAUTION]
> Lint and comment your changes.
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
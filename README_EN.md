# M2 Mac Genesis Robot Simulation Environment Setup Guide

## 1. Install Homebrew

Homebrew is a package manager for macOS. Run the following command in the Terminal:

```bash
/bin/bash -c “$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)”
```

Initialize the environment after installation:

```bash
echo 'eval “$(/opt/homebrew/bin/brew shellenv)”' >> ~/.zprofile
eval “$(/opt/homebrew/bin/brew shellenv)”
```

## 2. Install Miniforge

Install the Conda distribution optimized for the Apple Silicon environment:

```bash
brew install miniforge
```

Initialize the shell (based on zsh):

```bash
conda init zsh
source ~/.zshrc
```

## 3. Set up a Python virtual environment

Create and activate a virtual environment based on Python 3.10:

```bash
conda create -n myenv python=3.10
conda activate myenv
```

## 4. Install FFmpeg

Install FFmpeg for video processing:

```bash
brew install ffmpeg
```

## 5. Install Genesis

Install the Genesis package in the virtual environment:

```bash
pip install genesis-world
```

## Verify the installation

To verify that Genesis is installed correctly, try running the following in the Python interpreter:

```python
import genesis
print(genesis.__version__)
```

## Notes.

* All commands must be executed in sequence
* The virtual environment name (myenv) can be changed to any name you like.
* If you encounter any errors during the installation process, check the logs of each step.

## Example code for basic arm training
* You can test the example.py uploaded above to run the robot arm reinforcement learning training code.
#!/usr/bin/env bash

echo "Creating Environment..."
echo;
conda create -n Self-Driving-Car python=2.7
echo "Activating Environment..."
echo;
source activate Self-Driving-Car
echo "Installing PyTorch 0.3.1..."
echo;
conda install pytorch==0.3.1 -c pytorch
echo "Installing Kivy..."
echo;
conda install -c conda-forge kivy
echo "Installing matplotlib"
echo;
pip install matplotlib
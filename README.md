This project is used to identify if a voice is generally conceived as either male or female.

## Setup
* Install conda
* Create conda environment: `conda create -n gender-rec`
* Activate conda: `conda activate gender-rec`
* Install pip requirements: `pip install -r requirements.txt`

## Usage
* Place your dataset in the `input` folder.
* Run `python run.py`.
* The results will be in the `output` folder.

## Using GPU in conda:
If you for some reason want to use your GPU instead of CPU:

```bash
conda install -c conda-forge cudatoolkit=11.8.0
python3 -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.12.*
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$CUDNN_PATH/lib' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
```

Project used as inspiration: https://github.com/x4nth055/gender-recognition-by-voice
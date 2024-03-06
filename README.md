# On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models

Implementation to reproduce the simulation results of 
>B. Fesl, B. Böck, F. Strasser, M. Baur, M. Joham, and W. Utschick, "On the Asymptotic Mean Square Error Optimality of Diffusion Probabilistic Models," 2024, arXiv preprint: 2403.02957. \
>Link to paper: https://arxiv.org/abs/2403.02957

## Load data
Load data and pre-trained models from 
https://syncandshare.lrz.de/getlink/fiAsDStAV6i5FFJHyfhrcY/ (Passcode: Diffusion2024)
and move it to the project's directory.

## Information about required packages
Down below is a list of mandatory and optional packages with their versions, if a specific version is required.
### Mandatory packages
* conda-build
* cudatoolkit=11.8
* numpy
* pytorch=2.0.1
* torchvision=0.15.2=py310_cu118 (only if working with image data)
* pytorch-cuda=11.8
* scikit-learn
* scipy
* tqdm
* pip
* pytorch-fid==0.3
* matplotlib

### Optional packages (for jupyter notebooks or special script options)
* seaborn
* ipykernel
* ray-core
* ray-tune
* ray-train
* ray-dashboard

## Common Usage 
1. Train and test a DPM-based denoiser on GPU
```shell
python dpm_denoiser.py -d cuda:0
```

2. Load pre-trained model and evaluate it
```shell
python load_and_eval_dpm.py -d cuda:0
```

3. Evaluate real-valued baselines (GMM-based CME and LS)
```shell
python baselines.py
```

4. Evaluate complex-valued baseline with audio-data (GMM-based CME and LS)
```shell
python audio_gmm.py
```

## Data Options
1. rand_gmm
2. MNIST_gmm
3. FASHION_MNIST_gmm
4. audio_gmm

## Licenses
The diffusion model architecture is based upon the code from https://github.com/lucidrains/denoising-diffusion-pytorch under the following license:
> MIT License
>
> Copyright (c) 2020 Phil Wang
>
>Permission is hereby granted, free of charge, to any person obtaining a copy
>of this software and associated documentation files (the "Software"), to deal
>in the Software without restriction, including without limitation the rights
>to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
>copies of the Software, and to permit persons to whom the Software is
>furnished to do so, subject to the following conditions:
>
>The above copyright notice and this permission notice shall be included in all
>copies or substantial portions of the Software.
>
>THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
>IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
>FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
>AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
>LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
>OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
>SOFTWARE.

The real-valued Gaussian mixture model implementation stems from  https://scikit-learn.org/stable/modules/mixture.html and is covered by the following license:
>BSD 3-Clause License
>
>Copyright (c) 2007-2023 The scikit-learn developers. All rights reserved.
>
>Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
>AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
>IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
>DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
>FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
>DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
>SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
>CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
>OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
>OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The complex-valued extension of the Gaussian mixture model implementation stems from https://github.com/benediktfesl/GMM_cplx and is covered by the following license:
>BSD 3-Clause License
>
>Copyright (c) 2023 Benedikt Fesl. All rights reserved.
>
>Redistribution and use in source and binary forms, with or without
>modification, are permitted provided that the following conditions are met:
>
>1. Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
>
>2. Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
>3. Neither the name of the copyright holder nor the names of its
>   contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
>THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
>AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
>IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
>DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
>FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
>DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
>SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
>CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
>OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
>OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## ClipFace: Text-guided Editing of Textured 3D Morphable Models<br><sub>Official PyTorch implementation of SIGGRPAH 2023 paper</sub>

![Teaser](./docs/teaser.gif)

**ClipFace: Text-guided Editing of Textured 3D Morphable Models**<br>
Shivangi Aneja, Justus Thies, Angela Dai, Matthias Niessner<br>
https://shivangi-aneja.github.io/projects/clipface <br>

Abstract: *We propose ClipFace, a novel self-supervised approach for text-guided editing of textured 3D morphable model of faces. Specifically, we employ user-friendly language prompts to enable control of the expressions as well as appearance of 3D faces. We leverage the geometric expressiveness of 3D morphable models, which inherently possess limited controllability and texture expressivity, and develop a self-supervised generative model to jointly synthesize expressive, textured, and articulated faces in 3D. We enable high-quality texture generation for 3D faces by adversarial self-supervised training, guided by differentiable rendering against collections of real RGB images. Controllable editing and manipulation are given by language prompts to adapt texture and expression of the 3D morphable model. To this end, we propose a neural network that predicts both texture and expression latent codes of the morphable model. Our model is trained in a self-supervised fashion by exploiting differentiable rendering and losses based on a pre-trained CLIP model. Once trained, our model jointly predicts face textures in UV-space, along with expression parameters to capture both geometry and texture changes in facial expressions in a single forward pass. We further show the applicability of our method to generate temporally changing textures for a given animation sequence.*

<br>

### <a id="section1">1. Getting started</a>

#### Pre-requisites
- Linux
- NVIDIA GPU + CUDA 11.4 
- Python 3.8

#### Installation
- Dependencies:  
It is recommended to install dependecies using `pip`
The dependencies for defining the environment are provided in `requirements.txt`. For differentiable rendering, we use [NvDiffrast](https://nvlabs.github.io/nvdiffrast/), which can also be installed via pip.

### <a id="section2">2. Pre-trained Models required for training ClipFace</a>
Please download these models, as they will be required for experiments.

| Path                                  | Description
|:--------------------------------------| :----------
| [FLAME](https://flame.is.tue.mpg.de/) | We use FLAME 3DMM in our experiments. FLAME takes as input shape, pose and expression blendshapes and predicts mesh vertices. We used the **FLAME 2020 generic model** for our experiments. Using any other FLAME model might lead to wrong mesh predictions for expression manipulation experiments. Please download the model from the official website by signing their user agreement. Copy the generic model as `data/flame/generic_model.pkl` and FLAME template as `data/flame/head_template.obj` in the project directory.
| [DECA](https://deca.is.tue.mpg.de/)                              | DECA model predicts FLAME parameters for an RGB image. This is used during training StyleGAN-based texture generator, is available for download [here](https://drive.google.com/file/d/1hIpapFDc0dWJMJQgFHgWcDTSmjUCEmzL/) This can be skipped you don't intend to train the texture generator and use our pre-trained texture generator. 


### <a id="section3">3. Training</a>

The code is well-documented and should be easy to follow.
* **Source Code:**   `$ git clone` this repo and install the dependencies from `requirements.txt`. The source code is implemented in PyTorch Lightning and differentiable rendering with NvDiffrast so familiarity with these is expected. 
* **Dataset:** We used FFHQ dataset to train our texture generator. This is publicly available [here](https://github.com/NVlabs/ffhq-dataset). All images are resized to 512 X 512 for our experiments.
* **Data Generation:** From the original FFHQ dataset (70,000 images), we first remove images with headwear and eyewear. This gives us a clean and filtered FFHQ dataset (~45,000 images), which we use to train our stylegan-based texture generator. We use DECA model to predict FLAME parameters for each image in this filtered dataset. We pre-compute these FLAME parameters prior to training the generator model. We then use FLAME to predict mesh vertices for each image. Finally, we render the mesh with texture maps generated using our generator using differentiable rendering. For real images, we mask out background and mouth interior using alpha masks extracted from DECA. We provide the filtered image list, alpha masks and FLAME parameters for the filtered dataset [here](#section4) for simplicity.  
* **Training**: Run the corresponding scripts depending on whether you want to train the texture generator or perform text-guided manipulation. The scripts for training are available in `trainer/` directory.
  - **Texture Generator:** We use StyleGAN2 generator with adaptive discriminator [StyleGAN-ADA](https://github.com/NVlabs/stylegan2-ada) to generate UV maps due to its faster convergence. To train, run the following command:
  ```.bash
  python -m trainer.trainer_stylegan.train_stylegan_ada_texture_patch
  ```
  - **Text-guided  Manipulation:** We perform text-guided manipulation on textures generated using our pre-trained texture generator trained above. We first pre-train the mapper networks to predict zero offsets before performing text-guided manipulation, checkpoints available [here](#section4).  Run these scripts to perform text-guided manipulation.
  ```.bash
  # To train only for texture manipulation
  python -m trainer.trainer_texture_expression.train_mlp_texture
  
  # To train for both texture and expression manipulation
  python -m trainer.trainer_texture_expression.train_mlp_texture_expression
  ```

* **Path Configuration:** The configuration for training texture generator is `configs/stylegan_ada.yaml` and for text-guided manipulation is `configs/clipface.yaml`. Please refer to these files to configure the data paths and model paths for training.
  - Refer to `configs/stylegan_ada.yaml` to define the necessary data paths and model paths for training texture generator. 
  - Refer to `configs/stylegan_ada_clip_mlp.py` to define the necessary data paths and model paths for training texture and expression mappers for text-guided manipulation. Update the text prompt for manipulation in this file, defined as `altered_prompt`.  
  
### <a id="section4">4. ClipFace Pretrained Models and Dataset Assets</a>

| Path                                                                                                          | Description
|:--------------------------------------------------------------------------------------------------------------| :----------
| [Filtered FFHQ Dataset](https://drive.google.com/drive/folders/17be2i3L7Eb1Tgmkb_dKMYDAmDs4m2CVe?usp=sharing) | Download the filenames of Filtered FFHQ dataset; alpha masks and FLAME-space mesh vertices predicted using DECA. This can be skipped if you don't intend to train the texture generator and use our pre-trained texture generator.
| [Texture Generator](https://drive.google.com/file/d/1R8PZfoPwe_u4GpzeQ_FvCpBbr50DjwP9/)                       | The pretrained texture generator to synthesize UV texture maps.
| [UV Texture Latent Codes](https://drive.google.com/file/d/1vzAxA_6HFkECRMPgumrvLJW2M3xAiGvI/)                 | The latent codes generated from texture generator used to train the text-guided mapper networks.
| [Text-Manipulation Assets](https://drive.google.com/drive/folders/1beR9YHErIkb5EtLk0rLbcDV8TfdmEPt1/)         | The flame parameters & vertices for a neutral template face, These will be used to perform clip-guided manipulation. Copy these to `data/clip/` directory.
| [Pretrained Mappers](https://drive.google.com/file/d/15GUI-v3vf8VsAFwbBPS3EEYFaAEl0GQP/)                      | Pretrained mappers to predict zero offsets for text-guided manipulation
| Pretrained Texture & Expression Manipulation Models                                                           | Pretrained ClipFace checkpoints for different texture and expression styles shown in paper. Texture manipulation models can be downloaded from [here](https://drive.google.com/drive/folders/1B-BOL2EzBNBpZOmJZY7Xwpm783S8PzJs/); and expression manipulation models can be downloaded from [here](https://drive.google.com/drive/folders/1fxBm59PQB1_3Mh11DZOBb3za0gqtYy_F/).

[//]: # (| Pretrained Video Manipulation Sequences                                                                       | Coming soon!)




### <a id="section5">5. Inference</a>

* **Evaluation:**  Once training is complete, then to evaluate, specify the path to the uv-map latent code & mapper model (Line 42-43) in the script files and evaluate.
```.bash
  # To evaluate only for texture manipulation
  python -m tests.test_mlp_texture
  
  # To evaluate for both texture and expression manipulation
  python -m tests.test_mlp_texture_expression
  ```


</br>

### Citation

If you find our dataset or paper useful for your research , please include the following citation:

```

@inproceedings{aneja2023clipface,
    author    = {Aneja, Shivangi and Thies, Justus and Dai, Angela and Nie{\ss}ner, Matthias},
    booktitle = {SIGGRAPH '23 Conference Proceedings},
    title     = {ClipFace: Text-guided Editing of Textured 3D Morphable Models},
    year      = {2023},
    doi       = {10.1145/3588432.3591566},
    url       = {https://shivangi-aneja.github.io/projects/clipface/},
    issn      = {979-8-4007-0159-7/23/08},
}
```

</br>

### Contact Us

If you have questions regarding the dataset or code, please email us at shivangi.aneja@tum.de. We will get back to you as soon as possible.






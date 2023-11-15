# SpatialLM: Grounding Spatial Relations in Text-Only Language Models
This is the official repository of SpatialLM, an approach to ground spatial relation in text-only Language Models. You can check all the details in the paper [TODO: add link].

![System Diagram](assets/system-diagram.jpg)

## Installation
You can install all the dependencies using the `requirements.txt` file provided in the repository:

```
pip install -r requirements.txt
```
## Usage
If you want to execute the spatial pretraining on the Synthetic Spatial Training Dataset (SSTD), you can execute the following command. Take into account that you will have to provide a validation file. You can either generate that validation file on the fly using the code in `datasets/coco_for_spatial_pretraining.py` or use one of the provided ones in the repository (`datasets/spatial_pretraining_valfiles`). In any case, to generate the training examples, you will also need either COCO object detection annotations or the annotations derived from an object detector (we use [VinVL](https://github.com/pzzhang/VinVL) ):
```python
python spatialpretrain.py --model bert --location_encoding token --batch_size 28 --accumulate_grad_batches 4 --precision 16 --run_name bert_spatialpt_vinvl_withlocation --max_steps 20000 --root path_to_obj_detection_files --spatial_val_file datasets/spatial_pretraining_valfiles/validation-vinvl-alldistractors-noattr.json
```
To train and evaluate different models on the Visual Spatial Reasoning dataset ([VSR](https://github.com/cambridgeltl/visual-spatial-reasoning)), run the following the command:
```python
python train_seq2seq.py --model bert --dataset vsr --vsr_variant random --root datasets/vsr_seq2seq_files --batch_size 28 --accumulate_grad_batches 2 --precision 16 --max_steps 20000 --locations --run_name bert_vsr_random_withlocation_noattr --output_path output --evaluate --predictions_filename bert_vsr_random_wothlocation_noattr.out --cktp path_to_checkpoint
```
You can use the `--cktp` argument to load any checkpoint (for example, the checkpoint produced in the previous spatial pretraining step).

## Cite
If you find this repository useful, please consider citing:
```bibtex
@inproceedings{gazkune2023spatiallm,
  title={Grounding Spatial Relations in Text-Only Language Models},
  author={Azkune, Gorka and Salaberria, Ander and Agirre, Eneko},
  booktitle={TBA},
  year={2023}
}
```



![image](https://github.com/ihebakermi10/MLops-Translation-Pipeline/assets/90511874/fd163b0c-5e9b-4a9b-b746-617d5c9c648a)
# MLops-Translation-Pipeline

This project aims to create an MLOps pipeline for a translation model, including implementation details, evaluation metrics, and fine-tuning processes. 

## Project Overview
This repository contains the code and resources for setting up an MLOps pipeline for a translation model. The project includes training, evaluation, and deployment steps with a focus on automation and reproducibility.

## Installation
To set up the environment, install the required dependencies using `requirements.txt`.

```sh
pip install -r requirements.txt
```

## Project Structure
- `train.py`: Script for training the model.
- `val.py`: Script for validating the model.
- `requirements.txt`: Contains all the dependencies required for the project.

## Model Hub
The pre-trained models and fine-tuned versions are available on Hugging Face:
[Model Hub](https://huggingface.co/ihebaker10)

## Evaluation
The evaluation of the model is performed using TensorBoard, with metrics including WER, CER, and BLEU scores.

### Sample TensorBoard Visualizations
![Training Loss](![image](https://github.com/ihebakermi10/MLops-Translation-Pipeline/assets/90511874/c2983263-387f-489e-8f17-1ab1e37a70f4))
![BLEU Evaluation![image](https://github.com/ihebakermi10/MLops-Translation-Pipeline/assets/90511874/f3533637-294f-4a8b-969d-8066d86ad80b))

## Fine-Tuning Process
The fine-tuning process involves the following steps:
1. **Data Preparation**: Load and preprocess the dataset.
2. **Model Setup**: Initialize the pre-trained model.
3. **Training**: Fine-tune the model on the specific dataset.
4. **Evaluation**: Assess the model performance using appropriate metrics.
5. **Deployment**: Upload the fine-tuned model to Hugging Face.

### Detailed Steps
1. **Data Preparation**:
    - Use the `datasets` library to load the data.
    - Perform tokenization and other preprocessing steps.

2. **Model Setup**:
    - Initialize the `transformers` model.
    - Set up the optimizer and learning rate scheduler.

3. **Training**:
    - Use `torch` to train the model.
    - Monitor training with `tensorboard`.

4. **Evaluation**:
    - Evaluate using `wer`, `cer`, and `bleu` metrics.
    - Visualize results in TensorBoard.

5. **Deployment**:
    - Save the model and push to Hugging Face hub.

## Running the Project
To validate the model, run the following command:

```sh
python val.py
```

## Author
This project is developed by Iheb Akermi.
```

You can now proceed with the following steps:

1. **Create the GitHub repository**:
   - Go to GitHub and create a new repository named `MLops-Translation-Pipeline`.
   - Initialize it with a README file (replace the content with the updated README.md provided above).

2. **Upload files**:
   - Upload `requirements.txt`, `train.py`, and `val.py` to the repository.

3. **Add TensorBoard Images**:
   - Add a folder named `images` and upload the sample images for evaluation (ensure the paths in the README match the actual paths).

If you need any further assistance, feel free to ask!

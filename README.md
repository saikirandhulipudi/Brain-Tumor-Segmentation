# Brain Tumor Segementation using U-Net and Flask
Get the BraTS2020 Dataset [here.](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation)

## Introduction
The project employs deep learning for the semantic segmentation of MRI scans, specifically targeting tumor regions. 
The goal is to precisely identify and separate tumor areas within the images, contributing to medical diagnosis.
Developed a straightforward website to serve as a user interface. 
## Process
1. **Preprocessing:**
   - The dataset is provided in nibabel (.nii) format.
   - Images were resized, and less informative slices of scans were discarded.
   - Data preprocessing is handled by the BratsDataset class, which inherits from tf.keras.utils.Sequence.
2. **Model:**
   - Utilized the [U-Net architecture](https://github.com/VeerendraKocherla/BrainTumorSegmentation-UNet-Flask/blob/main/U-Net%20model%20architecture.txt) with nearly 31 million parameters for the task.
   - Its distinctive U-shaped structure includes an encoder to capture context and a decoder for precise localization and residual connections keep track of the original image during decoding.
   - This deep convolution neural network has been widely adopted in medical imaging due to its effectiveness in segmenting structures of interest from noisy or complex backgrounds.
3. **Training:**
    - Trained the model for 150 epochs (approx.), implementing early stopping with a patience of 5 epochs.
    - Used categorical crossentropy loss as criterion and manually adjusted learning rate of Adam optimizer for every 20 epochs.
    - Implemented checkpointing to save the optimal weights and accommodate potential failures. Utilized a Kaggle kernel for execution.
4. **Evaluation:**
    - Attained a cross-entropy loss of 0.015 and an accuracy of 99.52% on the testset.
    - Achieved mean Intersection over Union (IoU) of 86.54% and Dice coefficient of 0.42 on the testset.
    - refer predictions section in [this](https://github.com/VeerendraKocherla/BrainTumorSegmentation-UNet-Flask/blob/main/2-D_U-Net.ipynb) for predictions made by the model on testset.
5. **WebApp:**
   - Developed a straightforward web application using Flask.
   - Users input relative paths of flair and t1ce images and select the desired tumor area for evaluation.
   - Upon clicking "Generate," the model's prediction for the chosen region is displayed.
## Future work:
  - Train the dataset using a 3D U-Net model that employs conv3d layers instead of conv2d. This approach enables simultaneous extraction of information from all slices, enhancing the model's understanding of the data.
  - Enhance the interactivity of the web app.
  - Explore alternative algorithms such as Autoencoders, Attention-based UNet, and other methods for experimentation.
## References:
  - "Convolutional Neural Networks" course by Andrew Ng on Coursera.
  - "Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition" book by Aurélien Géron.
  - "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Ronneberger et al.







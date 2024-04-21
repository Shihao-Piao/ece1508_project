# ece1508_project
The target problem of our project is to conduct style transfer on original images, transform them into images that emulate the Ghibli style, and evaluate our model's performance in achieving this artistic translation.
### A. Data Collection and Baseline
Data Collection: We employed a web crawler to systematically gather images from the internet as our dataset for our project.
Baseline (NST): As a foundational step, we established Neural Style Transfer (NST) as our baseline model.
### B. Designing the Generator and Discriminator
CycleGAN Architecture: Moving beyond the baseline, we implemented the CycleGAN model, focusing on the Generator and the Discriminator. The Generator's role is to transform images from one domain (real-world landscapes) to another (Ghibli style), and vice versa.
### C. Loss Functions
Our training process incorporates three key loss functions to refine the model's performance: Adversarial Loss， Cycle Consistency Loss, and Identity Loss.
### D. Model Evaluation
Numerical Evaluation: For quantitative analysis, we utilized the Fréchet Inception Distance (FID) metric to measure the similarity between generated images and real images within each domain.
Manual Evaluation: Alongside numerical testing, we conducted manual evaluations of the generated images.

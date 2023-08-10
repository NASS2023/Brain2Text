# Brain2Text: An AI Model That Maps Brainwaves to Text

![](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/brain(new).gif)

# Abstract:
In today's rapidly evolving technological landscape, there exists a profound need to bridge the communication gap for indivisuals who face speech impediments. There are many people in the society who struggle to communicate, face challenges in expressing thoughts, and finds it difficult to engage in conversion due to aphasia. Our project is a contribution to help these people have communication with ease. 

It is well established that as a person thinks then corresponding to their thoughts, specific brainwaves are generated. These brainwaves follow a particular pattern and they carry an essence of the thoughts. This phenomenon has inspired us to extract and decipher these brainwaves in order to  decode the intended message. Hence this project offers a convergence of neuroscience and aritificial intelligence in order to bridge the communication gap.

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/waves.jpg)

The fundamental ambition of this project is to harness the capabilities of AI techniques to empower individuals with speech impediments to communicate effortlessly. The ultimate goal is to provide these people with some means of communication. 

The project is divided into three portions: MEG2VEC, TEXT2VEC and BRAIN2TEXT. MEG2VEC corresponds to converting Magnetoencephalography (MEG) signals to a vectorized form. TEXT2VEC corresponds to converting text to vectorized form. BRAIN2TEXT corresponds to the mapping of MEG vectors to text vectors. To solve these three steps, we have used CNN from the oneDNN library for MEG2VEC, Roberta for TEXT2VEC and a novel decoder architecture for BRAIN2TEXT.

# Inspiration:

University of California, San Francisco has started this study and their paper has been published in The New England Journal of Medicine: 

https://developer.nvidia.com/blog/transforming-brain-waves-into-words-with-ai/
https://www.nejm.org/doi/full/10.1056/NEJMoa2027540?query=featured_home

Recently Meta has also started working on this and a very recent paper from them is: 

https://hal.science/hal-03808317/file/BrainMagick_Neurips%20%283%29.pdf

They have simply found out a correlation between audio waves and brainwaves.

# Our Contributions to the Society:

✅ A contribution to speech impaired people of the society: This project provides a groundbreaking solution that leverage deep learning techniques to to bridge the communication gap for individuals with speed impediments and providing them an alternative means of expressing themselves. This project could ease the frustration and emotional burden often experienced by speech impaired individuals. This will enable people suffering from paralysis, aphasia or any speech impediments to have communication with ease.

✅ A contribution to medical research: This project would be a huge advancement in the field of neuroscience and linguistics using AI. This will help doctors and medical researchers gain new insights into the relationships between brain wave patterns and linguistic expressions.

✅ A contribution to mental health: This project can be used to read minds of people suffering from Dissociative Identity Disorder, Post Trauma Stress Disorder, and Sleep Disorders and potentially finding a cure to such diseases.

✅ A contribution to education: This project also contributes to education since this will enable talented professors with communication disabilities to deliver lectures much better. This will also enhance educational experiences for students with disabilities, fostering better engagement and learning.

✅ Law Enforcement and Forensic Research: In forensic science, this project can potentially assist law enforcement agencies in understanding the thought processes and intentions of suspected criminals.

✅ Medical Care: This project can be used to decipher brainwaves of comatose patients and medical caregivers would gain better insights into their patients cognitive state hence establish a form of rudimentary communication with the patient resulting in better care giving.

# Challenges:

1. Lack of medical domain knowledge: Being students of data science, we do not have much medical domain knowledge. Without this domain knowledge, it can be challenging to properly interpret medical data and draw meaningful insights.

2. Data Collection: Let's be realistic, we do not have access to MEG/EEG machines hence curating a perfect dataset to fit the task was a challenge.

3. Unavailability of dataset: Medical datasets are often not publicly available because of privacy policies and ethical considerations.

4. Limited Inspirations from prior work: This work of genetating text from brainwaves has not been explored by lot of researchers hence there was no benchmarks or well-established approaches available to draw inspiration from. The lack of precedent made it harder to devise the work.

Despite all these challenges, we have successfully generated text from brainwaves. Although we have not achieve a huge accuracy but we have definitely touched the tip of the iceberg. We have also established that, this task is possible to solve given proper medical guidance and data. 

# Our Work:

Lets begin with the dataset. We have obtained the dataset from:

https://osf.io/ag3kj/

This dataset has been curated from an experiment on 27 subjects. Each subject has 2 sessions. In each session, subjects were made to listen to 4 different stories. While the subjects were listening and thinking about the stories, their MEG signals were recorded. This implies that the MEG signals corresponds to the stories. There was 20 sensors applied on the subjects, hence 20 MEG signals were generated corresponding to each story. A sample from the dataset:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/denoise.jpg)

We have performed certain preprocessing and visualizations and have obtained the below waves. Under preprocessing we have performed:

1. Filtering: We have taken frequencies between 0.5Hz to 30Hz. This is in order to eleminate Gamma Waves since Gamma Waves is not associated with thought processes.

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/gammawaves.jpg)

2. Independent Component Analysis: Independent component analysis (ICA) is a statistical technique used for signal processing to seperate multivariate signal into additive and independent components. The technique is an effective method for removing artifacts and separating sources of the brain signals from these recordings.

3. Cropping

4. Resizing

5. Normalization

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/img.jpg)

Our aim here is to extract information from these images and generate the corresponding story. Although story generation is not the aim of this project but due to lack of dataset, we had to narrow down our task. However, this projects makes a strong statement that it is possible to generate text from brainwaves. Hence, the input to the architecture is a MEG images and output of the architecture is a story.

Now, the architecture has three stages:

# MEG2VEC: Generating vectors from MEG Images

We have used the CNN architecture present in oneDNN library to extract features from the MEG images. 

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/cnn2.png)

The architecture: 
--> Convolutional 2D (3,32)
--> Batch Normalization + ReLu + Dropout (Rate=0.1)
--> Convolutional 2C (32,60)
--> Batch Normalization + ReLu + Dropout (Rate=0.1).
--> Max Pooling
--> Linear Layer (16 * 144 * 144,512)
--> Linear Layer (512)

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/cnn_arch.jpg)

With this architecture, we have classified the brainwaves to its respective stories. In this phase, the classification accuracy is:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/cnn_results.jpg)

# TEXT2VEC: Generating vectors from text stories

We have used pre-trained model Roberta for encoding our texts into embeddings and again decoding the embeddings to get the text back.

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/Screenshot%202023-08-10%20221705.png)

# BRAIN2TEXT:


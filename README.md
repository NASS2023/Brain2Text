# Though2Text: An AI model that maps brainwaves to text

https://github.com/NASS2023/Brain2Text/assets/141930141/6b6b4c24-a5f9-4f3a-979e-75e20dcd8a4d

This is our website interface hosted in local machine using a Flask App. This app is not uploaded on Github because of size constraints.

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

✅ **A contribution to speech impaired people of the society**: This project provides a groundbreaking solution that leverage deep learning techniques to bridge the communication gap for individuals with speed impediments and providing them an alternative means of expressing themselves. This project could ease the frustration and emotional burden often experienced by speech impaired individuals. This will enable people suffering from paralysis, aphasia or any speech impediments to have communication with ease.

✅ **A contribution to medical research**: This project would be a huge advancement in the field of neuroscience and linguistics using AI. This will help doctors and medical researchers gain new insights into the relationships between brain wave patterns and linguistic expressions.

✅ **A contribution to mental health**: This project can be used to read minds of people suffering from Dissociative Identity Disorder, Post Trauma Stress Disorder, and Sleep Disorders and potentially finding a cure to such diseases.

✅ **A contribution to education**: This project also contributes to education since this will enable talented professors with communication disabilities to deliver lectures much better. This will also enhance educational experiences for students with disabilities, fostering better engagement and learning.

✅ **Law Enforcement and Forensic Research**: In forensic science, this project can potentially assist law enforcement agencies in understanding the thought processes and intentions of suspected criminals.

✅ **Medical Care**: This project can be used to decipher brainwaves of comatose patients which will help medical caregivers to gain better insights into their patients cognitive state and hence establish a form of rudimentary communication with the patient resulting in better care giving.

# Challenges:

1. **Lack of medical domain knowledge**: Being students of data science, we do not have much medical domain knowledge. Without this domain knowledge, it can be challenging to properly interpret medical data and draw meaningful insights.

2. **Data Collection**: Let's be realistic, we do not have access to MEG/EEG machines hence curating a perfect dataset to fit the task was a challenge.

3. **Unavailability of dataset**: Medical datasets are often not publicly available because of privacy policies and ethical considerations.

4. **Limited Inspirations from prior work**: This work of genetating text from brainwaves has not been explored by lot of researchers hence there was no benchmarks or well-established approaches available to draw inspiration from. The lack of precedent made it harder to devise the work.

Despite all these challenges, we have successfully generated text from brainwaves. Although we have not achieve a huge accuracy but we have definitely touched the tip of the iceberg. We have also established that, this task is possible to solve given proper medical guidance and data. 

# Our Work:

## Dataset: 

Lets begin with the dataset. We have obtained the dataset from:

https://osf.io/ag3kj/

This dataset has been curated from an experiment on 27 subjects. Each subject has 2 sessions. In each session, subjects were made to listen to 4 different stories. While the subjects were listening and thinking about the stories, their MEG signals were recorded. This implies that the MEG signals corresponds to the stories. There was 20 sensors applied on the subjects, hence 20 MEG signals were generated corresponding to each story. A sample from the dataset:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/denoise.jpg)

This image corresponds to one person listening to one story during one session. Our dataset has only 196 such images, since few subjects did not turn up for second session. Each images are of size 576 * 576, which is again very small in size.

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

## MEG2VEC: Generating vectors from MEG Images

This is the **phase 1** of the task. We have used the CNN architecture present in oneDNN library to extract features from the MEG images.

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/onednn.jpg)

oneDNN is an Intel oneAPI Deep Neural Network Library which provides us with implementation of optimised deep learning blocks. We used this framework to improve the performance of our CNN model and in turn extract the embeddings corresponding to the MEG signals. oneDNN gives us the opportunity to run our code on CPU or GPU as per our convinience which greatly enhanced our training speed.

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

## TEXT2VEC: Generating vectors from text stories

This is the **phase 2** of the task. We have used pre-trained Roberta-Large for encoding and decoding texts.

Steps to encode:

1. Tokenize the data

2. Pass the tokenized data to the model

3. Extract the last hidden layer output, this is our embedding of the tokens

4. Average pooling on embeddings

5. Mapping the index ids of the tokens with the embedded vector

6. Create a dictionary having tokens and its corresponding embedding

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/textvec.jpg)

## BRAIN2TEXT:

This is the **phase 3** of the task. Here we take the outputs of the previous two phases and try to set up a mapping between them. We have build a **novel decoder** architecture for the text generation purpose. Our novel architecture is as follows:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/decoder.jpg)

This is the general architecture of a decoder:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/decoderarch.jpg)

To this decoder architecture, we give the wave embeddings as input and the text embeddings are the output. The model has been able to generate stories with more than 58% F-Measure (Rouge 1). With so less amount of data, achieving these scores is a huge success. The Rouge Scores are as follows:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/rouge.jpg)

Rouge Score lies between 0 and 1. The closer the value to 1, more is the similarity between actual and generated sentences. Rouge-1 considers 1-gram, Rouge-2 considers 2-gram and Rouge-L considers the longest common subsequence. The formula are as follow:

![image](https://github.com/NASS2023/Brain2Text/blob/main/IMAGES/rougeformula.jpg)

# Future Prospect:

The future idea of our project is to create a wearable device that resembles headphones. The core functionality of the device involves reading brainwave, converting them into text and using a voice assistant to read the text. Developing the hardware for such a wearable device involves several challenges. It requires expertise in sensor technology and electronics. Partnering with companies experienced in hardware architecture and wearable technology can accelerate the development process. These companies can bring their knowledge of designing compact and functional hardware components to enhance this project. 

To ensure the medical effectiveness and safety of our device, collaboration with medical professionals is crucial. Neuroscientists and medical researchers can provide guidance on the interpretation of brainwave and the relevance of the results obtained. Their expertise can help us refine our device's algorithms and make sure that the device is safe to wear. The potential impact of this project goes beyond medical research. If successful, the brainwave-reading wearable could have applications in various fields such as education, communication, and assistive technology. It will provide indivisuals having speech impediments with some means of having communication.

# Limitation:

1. Due to super time constraint, we are unable to build a tokenizer of our own. Although, we have a novel decoder architecture, but the architecture is using pre-trained embeddings from Roberta. Due to this reason, our outputs are specific to Roberta style of embeddings which is not always meaningful to the end user. To resolve this issue, we have to build a tokenizer of our own.

2. Since we have brainwaves corresponding to long stories, each datapoint is huge in length. Models find it difficult to learn from datapoints having long seperated dependencies. Rather, if we had conversational data and their corresponding waves, then model learning would have been far better since the datapoints are shorter in length.

# Steps for improvement:

1. **Curating a proper dataset**: The model that performs thoughts extraction from brainwaves will perform best if trained with conversational data and their corresponding brainwaves. This is because conversational data are shorter in length and hence shorter waves would be generated. Lerning patterns from shorter waves and shorter texts would be much easier to the model.

2. **Proper Data Type**: Instead of doing this task based on MEG images, we can obtain better performance if performed on MEG readings. This is because, we definitely loose some amount of information while extracting features from images. Hence, having the MEG reading would perform better since they are raw numerical data. There would be no need to perform information extraction.

3. **Medical Guidance**: Collaborating with doctors and medical researchers is of utmost importance for the success of this project.

4. **Segmentation**: In case we have to deal with MEG images only, we can have an additional step in the preprocessing of images i.e., segmentation. Instead of having 20 MEG waves in one image, we can segment the image into 20 parts, each part having an indivisual image. This step would help CNN focus on indivisual waves and extract better information with minimal loss. Due to time contraints, we unable to perform this step but we would definitely do it in the future.

5. **Tokenizer**: Having a corpus of our own and developing a tokenizer from scratch is very important for smooth encoding and perfect decoding of texts. Again due to time constraints, we were unable to perform this step.


# My Learning from oneAPI:

✅ Using IDC (Intel Dev Cloud): This platform has provided us with cloud which we can easily connect with our localhost and use the GPUs present in the servers. This has helped us train our models really fast hence saving our time and allowing us to focus more on improving our work.

✅ Using oneDNN: This library contains optimised versions of neural networks like ANN, RNN and CNN. Since we had to extract features from images, we have used the CNN architecture supported by this library. This has helped us train our models really fast hence reducing time complexity of the task.

✅ Using DevMesh: This is a platform that provides us with a space to showcase our work and share our work with many other developers present in the network. This helps us connect with other developers, learn from them and expand our networking.

# Presentation Video:

https://drive.google.com/file/d/1-9TyYi_Dqnh6FamdC40O3UBnnnhfQ-3A/view


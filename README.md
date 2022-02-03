

## Abstract

Reflections often degrade the quality of images by obstructing the background scenes. This is not desirable for everyday users, and it negatively impacts the performance of multimedia applications that process images with reflections. Most current methods for removing reflections utilize supervised learning models. These models require an extensive number of image pairs of the same scenes with and without reflections to perform well. However, collecting such image pairs is challenging and costly. Thus, most current supervised models are trained on small datasets that cannot cover the numerous possibilities of real-life images with reflections.  In this paper, we propose an unsupervised method for single-image reflection removal. Instead of learning from a large dataset, we optimize the parameters of two cross-coupled deep convolutional neural networks on a target image to generate two exclusive background and reflection layers. In particular, we design a network model that embeds semantic features extracted from the input image and utilizes these features in the separation of the background layer from the reflection layer. We show through objective and subjective studies on benchmark datasets that the proposed method substantially outperforms current methods in the literature. The proposed method does not require large datasets for training, removes reflections from individual single images, and does not impose constraints or assumptions on the input images. 











## Get Started 

1. Install "Python" with the version >= 3.5 
    In case of using RGC Module at SFU, please load Python module 
    ```
    module load LANG/PYTHON/3.5.2-SYSTEM
    ```


2. Install all the requirements (Linux) 
    ```
    python3 -m venv venv 
    source venv/bin/activate 
    python3 -m pip install -r requirements.txt 
    ```
    if you would get some errors, please update pip version  



3. After setting up the data and the model parameters in the separator (the model), estimated background and reflections are visible through the output directory.
	```
    python separator3_2.py 
    ```


4. Using "mask" python script, regions of reflection are achieved from the estimated layers from the previous steps.



5. For the refinment module, we will use this achieved mask, either iteratively by giving separationg it into 4 parts and refining the input in 4 steps, or by using the whole mask in 1 step, following the steps below :


	- Code and model are downloaded through 
	```
	git clone --single-branch https://github.com/zengxianyu/crfill
	```

	- Install dependencies:
	```
	conda env create -f environment.yml
	```
	or manually install these packages in a Python 3.6 enviroment: 

	```pytorch=1.3.1```, ```opencv=3.4.2```, ```tqdm```


	Use the code:

	with GPU:
	```
	python test.py --image path/to/images --mask path/to/masks --output path/to/save/results
	```
	without GPU:
	```
	python test.py --image path/to/images --mask path/to/masks --output path/to/save/results --nogpu
	```
	```path/to/images``` is the path to the folder of input images; ```path/to/masks``` is the path to the folder of the masks; ```path/to/save/results``` is where the results will be saved. 



	:mega: :mega: The white area of a mask should cover all pixels in the reflection regions. :mega: :mega:


	


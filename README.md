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



3. After setting up the data and the model parameters in the separator, estimated background and reflections are visible through the output directory.



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


	


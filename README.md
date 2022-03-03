# PyramidBox-A-context-Assisted-Single-Shot-Face-Detector-
Python based implementation of PyramidBox model in Google Colab 


REQUIREMENTS:
numpy
torch==0.4.1.post2
torchvision==0.2.1
easydict
opencv

DOWNLOADING DATA:
	•	Download the DSFD framework through the link given below and upload it to the google drive :
https://github.com/yxlijun/Pyramidbox.pytorch
	•	 Download the PyramidBox pretrained model from the link given below:
https://drive.google.com/file/d/1jLHIwN15u73qr-8rmthZEZWQfnAq6N9C/view
	•	Download the Widerface Dataset along with the face annotations from the following link: http://shuoyang1213.me/WIDERFACE/
	•	Download the Evaluation Code and Validation Results from the following link: http://shuoyang1213.me/WIDERFACE/
	•	Download the following git repository and only keep widerface_evaluate folder and delete all other files and folders:
https://github.com/peteryuX/retinaface-tf2#Models


STEPS to follow:
	•	Moving files to directories:

	•	Make a folder name weights in Pyramidbox.pytorch-master and put the downloaded pretrained model inside that.
	•	Put the eval_tools folder(Evaluation Code and Validation Results) in Pyramidbox.pytorch-master and from that move the file name as wider_face_val.mat present in ground_truth folder to main eval_tools folder.
	•	Put the widerface_evaluate folder in Pyramidbox.pytorch-master.
	•	Move wider_test.py from tools folder to the main Pyramidbox.pytorch-master.
	•	 Make a folder name datasets in google drive in Pyramidbox.pytorch-master and inside datasets folder make another folder name WIDER and put all the downloaded Widerface Dataset along with face annotations inside that.


	•	Changes in specific directories and files:
	
Follow Steps Given In Changes.py file
	

RESULTS

![Pyramidbox photo1](https://user-images.githubusercontent.com/87081613/156588770-a1c2c930-ed2e-479a-bec2-a861f22bd3e5.jpeg)
![PyramidBox Photo 2](https://user-images.githubusercontent.com/87081613/156589414-9e04cbd3-0697-4e6b-8ef6-7e606a3de099.jpeg)
![PyramidBox Photo 3](https://user-images.githubusercontent.com/87081613/156589453-1878d22a-1e16-48a7-9072-87802de276f9.jpeg)
![PyramidBox Photo 4](https://user-images.githubusercontent.com/87081613/156589482-1e3ce9f2-93f6-4f36-92cf-53fedb456921.jpeg)

PRECISION Vs RECALL CURVE

![PyramidBox Photo 5](https://user-images.githubusercontent.com/87081613/156589620-2a690d8a-c5c0-4b70-8288-cba92c6923fb.jpeg)

 
       











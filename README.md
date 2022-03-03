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

(DATA PREPARATION)

	•	prepare_wider_data.py
replace xrange with range in line no. 49,59 and 72
	•	data/config.py
line 64- _C.HOME = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/datasets'
line 70- _C.FACE.WIDER_DIR = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/datasets/WIDER'
RUN THE FOLLOWING SETS OF CODE IN GOOGLE COLAB:
(i) from google.colab import drive
drive.mount("/content/drive", force_remount=True)
(ii) %cd '/content/drive/MyDrive/Pyramidbox.pytorch-master'
(iii) !python prepare_wider_data.py (this will generate two files in data folder) 

line 68- _C.FACE.TRAIN_FILE = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/face_train.txt'
line 69- _C.FACE.VAL_FILE = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/face_val.txt'

           (RUNNING THE DEMO FILE)

	•	demo.py
line 29- default='/content/drive/MyDrive/Pyramidbox.pytorch-master/weights/pyramidbox_120000_99.02.pth'
	•	layers/functions/_init_.py
line 2- replace detection with .detection

RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
(i) !python demo.py

            (RUNNING THE TEST FILE)

	•	wider_test.py
line 30-
default='/content/drive/MyDrive/Pyramidbox.pytorch-master/weights/pyramidbox_120000_99.02.pth'
line 164-
​​wider_face = sio.loadmat(
           '/content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/wider_face_val.mat')
line 232- replace xrange with range 
line 229-
fout.write('{:s}\n'.format(event[0][0] + '/' + im_name.decode('utf-8') + '.jpg'))
line 227-
fout = open(osp.join(bytes(save_path, 'utf-8'), event[0][0].encode('utf-8'), im_name + bytes('.txt', 'utf-8')), 'w')
line 215- add code given below, after line 215 and correct the indentation:
with torch.no_grad():
line 199-
in_file = os.path.join(bytes(imgs_path, 'utf-8'), event[0][0].encode('utf-8'), im_name[:] + bytes('.jpg','utf-8'))
line 193-
path = os.path.join(bytes(save_path, 'utf-8'), event[0][0].encode('utf-8'))
line 156-
try:
  dets = dets[0:750,:]
except:
  dets = np.zeros((1,5))
	•	layers/functions/detection.py
line 60-
if scores.dim() == 0 or scores.numel() == 0:

RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
(i) !python wider_test.py

            (PLOTTING THE PRECISION Vs RECALL GRAPH)

(ii) %cd '/content/drive/MyDrive/Pyramidbox.pytorch-master/widerface_evaluate'
(iii) !python3 setup.py build_ext --inplace


	•	widerface_evaluate/evaluation.py
line 282- add following lines of code:
plt.plot(recall, propose, color = 'tab:blue')
plt.savefig("Final Graph.png",bbox_inches="tight")
plt.title('Precision Vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
line 15- add following line of code:
import matplotlib.pyplot as plt

RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
(i) !python3 evaluation.py -p /content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/pyramidbox_val -g /content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/ground_truth

The above line of code will create the required graph in the widerface_evaluate folder. 













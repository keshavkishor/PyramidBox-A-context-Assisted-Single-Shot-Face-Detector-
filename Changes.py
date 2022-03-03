#Changes in specific directories and files:

# (DATA PREPARATION)

	•	prepare_wider_data.py
  
# replace xrange with range in line no. 49,59 and 72

#	•	data/config.py
# line 64- 
_C.HOME = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/datasets'
# line 70- 
_C.FACE.WIDER_DIR = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/datasets/WIDER'

# RUN THE FOLLOWING SETS OF CODE IN GOOGLE COLAB:------------------------------------------------------------------------------------------------------------
from google.colab import drive
drive.mount("/content/drive", force_remount=True)
%cd '/content/drive/MyDrive/Pyramidbox.pytorch-master'
!python prepare_wider_data.py #(this will generate two files in data folder) 

# line 68- 
_C.FACE.TRAIN_FILE = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/face_train.txt'
# line 69- 
_C.FACE.VAL_FILE = '/content/drive/MyDrive/Pyramidbox.pytorch-master/data/face_val.txt'

#  (RUNNING THE DEMO FILE)--------------------------------------------------------------------------------------------------------------------------------------------

# •	demo.py
# line 29- 
default='/content/drive/MyDrive/Pyramidbox.pytorch-master/weights/pyramidbox_120000_99.02.pth'
# •	layers/functions/_init_.py
# line 2- replace detection with .detection

# RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
 !python demo.py

#  (RUNNING THE TEST FILE)------------------------------------------------------------------------------------------------------------------------------------------

#	•	wider_test.py
# line 30-
default='/content/drive/MyDrive/Pyramidbox.pytorch-master/weights/pyramidbox_120000_99.02.pth'
# line 164-
​​wider_face = sio.loadmat(
           '/content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/wider_face_val.mat')
# line 232- replace xrange with range 
# line 229-
fout.write('{:s}\n'.format(event[0][0] + '/' + im_name.decode('utf-8') + '.jpg'))
# line 227-
fout = open(osp.join(bytes(save_path, 'utf-8'), event[0][0].encode('utf-8'), im_name + bytes('.txt', 'utf-8')), 'w')
# line 215- add code given below, after line 215 and correct the indentation:
with torch.no_grad():
# line 199-
in_file = os.path.join(bytes(imgs_path, 'utf-8'), event[0][0].encode('utf-8'), im_name[:] + bytes('.jpg','utf-8'))
# line 193-
path = os.path.join(bytes(save_path, 'utf-8'), event[0][0].encode('utf-8'))
# line 156-
try:
  dets = dets[0:750,:]
except:
  dets = np.zeros((1,5))
#	•	layers/functions/detection.py
# line 60-
if scores.dim() == 0 or scores.numel() == 0:

# RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
(i) !python wider_test.py

#   (PLOTTING THE PRECISION Vs RECALL GRAPH)--------------------------------------------------------------------------------------------------------------------------

(ii) %cd '/content/drive/MyDrive/Pyramidbox.pytorch-master/widerface_evaluate'
(iii) !python3 setup.py build_ext --inplace


# •	widerface_evaluate/evaluation.py
# line 282- add following lines of code:
plt.plot(recall, propose, color = 'tab:blue')
plt.savefig("Final Graph.png",bbox_inches="tight")
plt.title('Precision Vs Recall')
plt.xlabel('Recall')
plt.ylabel('Precision')
# line 15- add following line of code:
import matplotlib.pyplot as plt

# RUN THE FOLLOWING SET OF CODE IN GOOGLE COLAB:
(i) !python3 evaluation.py -p /content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/pyramidbox_val -g /content/drive/MyDrive/Pyramidbox.pytorch-master/eval_tools/ground_truth

# The above line of code will create the required graph in the widerface_evaluate folder. 


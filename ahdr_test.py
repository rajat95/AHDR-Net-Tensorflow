	
import os,time,cv2,sys,math
import tensorflow as tf
import numpy as np
from AHDR import build_ahdr


def radiance_writer(out_path, image):
	with open(out_path, "wb") as f:
		f.write(bytes("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n",'UTF-8'))
		f.write(bytes("-Y %d +X %d\n" %(image.shape[0], image.shape[1]),'UTF-8'))
		brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
		mantissa = np.zeros_like(brightest)
		exponent = np.zeros_like(brightest)
		np.frexp(brightest, mantissa, exponent)
		scaled_mantissa = mantissa * 255.0 / brightest
		rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
		rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
		rgbe[...,3] = np.around(exponent + 128)
		rgbe.flatten().tofile(f)


def log_tonemap_output(im):
   return np.log(1+5000.0*im)/np.log(1+5000.0)



def ldr_to_hdr(im,bias,gamma = 2.2):

	exp_bias = bias
	exp_time = 2**exp_bias
	im_out = im**gamma
	im_out = im_out/exp_time
	return im_out


def image_as_ubyte(im):
	return (255*im).astype(np.uint8)



#define inputs and outputs
le_inp_pl = tf.placeholder(tf.float32,shape=[None,None,None,6])
me_inp_pl = tf.placeholder(tf.float32,shape=[None,None,None,6])
he_inp_pl = tf.placeholder(tf.float32,shape=[None,None,None,6])

input_exposure_stacks = [le_inp_pl,me_inp_pl,he_inp_pl]



print('Loading the model file')

network = build_ahdr([le_inp_pl,me_inp_pl,he_inp_pl])


config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.Session(config=config)


print('Loading the model weights from checkpoint')

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())	
model_checkpoint_name = os.path.join(os.path.join('checkpoint','0035'),"model.ckpt")
saver.restore(sess, model_checkpoint_name)



exposure_folder = os.path.join('test_images','scene_3')
ex_bias_path = os.path.join(exposure_folder,'ExposureBias.txt')
with open(ex_bias_path) as f:
		exp_info = [ float(float(v.rstrip('\n\r'))) for v in f.readlines()]
		
ev0 = exp_info[0]	
ev1 = exp_info[1]
ev2 = exp_info[2]

cv2_img_he = cv2.imread(os.path.join(exposure_folder,'img_3_aligned.tif'),-1)
h,w = cv2_img_he.shape[:2]			
input_image_he = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_he,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
input_image_he_gamma= ldr_to_hdr(input_image_he,ev2)
input_image_he_c = np.concatenate([input_image_he,input_image_he_gamma],axis=3)


cv2_img_me = cv2.imread(os.path.join(exposure_folder,'img_2_aligned.tif'),-1)
h,w = cv2_img_me.shape[:2]
input_image_me = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_me,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
input_image_me_gamma= ldr_to_hdr(input_image_me,ev1)
input_image_me_c = np.concatenate([input_image_me,input_image_me_gamma],axis=3)

cv2_img_le = cv2.imread(os.path.join(exposure_folder,'img_1_aligned.tif'),-1)
h,w = cv2_img_le.shape[:2]
input_image_le = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_le,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
input_image_le_gamma = ldr_to_hdr(input_image_le,ev0)
input_image_le_c = np.concatenate([input_image_le,input_image_le_gamma],axis=3)



output_image_pred = sess.run([network],\
							feed_dict={input_exposure_stacks[0]:input_image_le_c, \
										input_exposure_stacks[1]:input_image_me_c,\
										input_exposure_stacks[2]:input_image_he_c,\
										})
output_image = np.squeeze(output_image_pred)
radiance_writer("scene_3_pred.hdr",output_image)
cv2.imwrite("scene_3_pred_tonemapped.png",image_as_ubyte(log_tonemap_output(cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))))


sess.close()
print('test completed for scene_3')




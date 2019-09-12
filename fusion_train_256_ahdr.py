	
from __future__ import absolute_import
from __future__ import print_function
import os,time,cv2,sys,math
import tensorflow as tf
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
from utils import utils, helpers
from builders import fusion_model_builder
import datetime



def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def radiance_writer(out_path, image):
	with open(out_path, "wb") as f:
		f.write(bytes("#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n",'UTF-8'))
		f.write(bytes("-Y %d +X %d\n" %(image.shape[0], image.shape[1]),'UTF-8'))
		brightest = np.max(image,axis=2)

		mantissa = np.zeros_like(brightest)
		exponent = np.zeros_like(brightest)	

		np.frexp(brightest, mantissa, exponent)
		scaled_mantissa = mantissa * 255.0 / brightest
		rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
		rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
		rgbe[...,3] = np.around(exponent + 128)
		rgbe.flatten().tofile(f)

def compute_psnr(img1, img2):
   mse = np.mean((img1-img2)**2)
   if mse == 0:
       return 100
   PIXEL_MAX = 1.0 # input -1~1
   return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def log_tonemap(im):
   return tf.log(1+5000*im)/tf.log(1+5000.0)


def log_tonemap_output(im):
   return np.log(1+5000*im)/np.log(1+5000.0)



parser = argparse.ArgumentParser()
parser.add_argument('--nTry', type=int, default=None, help='Current try number')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train for')
parser.add_argument('--id_str', type=str, default="", help='Unique ID string to identify current try')
parser.add_argument('--status_id', type=int, default=1, help='Status ID to write to status.txt. Can be 1, 2 or 3')
parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--image', type=str, default=None, help='The image you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="hdr_ddg_dataset_ulti_13thJuly", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=256, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=256, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=16, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=100000, help='The number of images to used for validations')
parser.add_argument('--model', type=str, default="DRIB_4_four_conv", help='The model you are using. See model_builder.py for supported models')
parser.add_argument('--frontend', type=str, default="ResNet101", help='The frontend you are using. See frontend_builder.py for supported models')
parser.add_argument('--save_logs', type=str2bool, default=True, help='Whether to save training info to the corresponding logs txt file')
parser.add_argument('--log_interval', type=int, default=100, help='Log Interval')
parser.add_argument('--init_lr', type=float, default=0.0002, help='Initial learning rate')
parser.add_argument('--lr_decay', type=float, default=0.94, help='Initial learning rate')
parser.add_argument('--loss', type=str, default='l2', help='Choose between l2 or l1 norm as a loss function')
parser.add_argument('--logdir', type=str, default='/workspace/logs', help='Choose between l2 or l1 norm as a loss function')
parser.add_argument('--crop_pixels_height',type=int,default=10,help='Location of input image')


args = parser.parse_args()
try_name = "Try%d_%s_%s"%(args.nTry,args.model,args.id_str)



if not os.path.isdir(try_name):
	os.makedirs(try_name)

if args.save_logs:
	if args.continue_training:
		log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
		status = open("status%d.txt"%(args.status_id),'a')
	else: 
		log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'w')
		status = open("status%d.txt"%(args.status_id),'w')

config = tf.ConfigProto()
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

if not os.path.exists(os.path.join(args.logdir,try_name,'train')):
    os.makedirs(os.path.join(args.logdir,try_name,'train'),exist_ok=True)

if not os.path.exists(os.path.join(args.logdir,try_name,'test')):
    os.makedirs(os.path.join(args.logdir,try_name,'test'),exist_ok=True)


train_writer = tf.summary.FileWriter('{}/{}/train'.format(args.logdir,try_name))
test_writer = tf.summary.FileWriter('{}/{}/test'.format(args.logdir,try_name))

train_loss_pl = tf.placeholder(tf.float32,shape=None)
train_loss_summary =tf.summary.scalar('train_loss',train_loss_pl)

test_loss_pl = tf.placeholder(tf.float32,shape=None)
test_loss_summary =tf.summary.scalar('test_loss',test_loss_pl)

test_psnr_pl = tf.placeholder(tf.float32,shape=None)
test_psnr_summary =tf.summary.scalar('val_psnr',test_psnr_pl)


le_image_pl = tf.placeholder(tf.float32,shape=[args.batch_size,args.crop_width,args.crop_height,3])
me_image_pl = tf.placeholder(tf.float32,shape=[args.batch_size,args.crop_width,args.crop_height,3])
he_image_pl = tf.placeholder(tf.float32,shape=[args.batch_size,args.crop_width,args.crop_height,3])
gt_image_pl = tf.placeholder(tf.float32,shape=[args.batch_size,args.crop_width,args.crop_height,3])


le_image_summ = tf.summary.image('le images',le_image_pl,max_outputs=args.batch_size)
me_image_summ = tf.summary.image('me images',me_image_pl,max_outputs=args.batch_size)
he_image_summ = tf.summary.image('he images',he_image_pl,max_outputs=args.batch_size)
gt_image_summ = tf.summary.image('gt images',gt_image_pl,max_outputs=args.batch_size)

input_exposure_stacks = [tf.placeholder(tf.float32,shape=[None,None,None,6]) for x in range(3)]
gt_exposure_stack = tf.placeholder(tf.float32,shape=[None,None,None,3])


lr = tf.placeholder("float", shape=[])
network, init_fn = fusion_model_builder.build_model(model_name=args.model, frontend=args.frontend, input_exposure_stack=input_exposure_stacks, crop_width=args.crop_width, crop_height=args.crop_height, is_training=True)



str_params = utils.count_params()
print(str_params)
if args.save_logs:
	log_file.write(str_params + "\n")


if args.loss == 'l2':
	loss = tf.losses.mean_squared_error(log_tonemap(gt_exposure_stack), log_tonemap(network))
elif args.loss == 'l1':
	loss = tf.losses.absolute_difference(log_tonemap(gt_exposure_stack), log_tonemap(network))

opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())	
train_writer.add_graph(sess.graph)

# Load a previous checkpoint if desired 
model_checkpoint_name = "%s/ckpts/latest_model_"%(try_name) + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training:
	print('Loading latest model checkpoint')
	saver.restore(sess, model_checkpoint_name)
	print('Loaded latest model checkpoint')




print("\n***** Begin training *****")
print("Try -->", args.nTry)
print("Dataset -->", args.dataset)
print("Model -->", args.model)
print("Crop Height -->", args.crop_height)
print("Crop Width -->", args.crop_width)
print("Num Epochs -->", args.num_epochs)
print("Batch Size -->", args.batch_size)
print("Save Logs -->", args.save_logs)

avg_loss_per_epoch = []
avg_val_loss_per_epoch = []
avg_psnr_per_epoch = []

if args.save_logs:
	log_file.write("\nDataset --> " + args.dataset)
	log_file.write("\nModel --> " + args.model)
	log_file.write("\nCrop Height -->" + str(args.crop_height))
	log_file.write("\nCrop Width -->" + str(args.crop_width))
	log_file.write("\nNum Epochs -->" + str(args.num_epochs))
	log_file.write("\nBatch Size -->" + str(args.batch_size))
	log_file.close()

	status.write("\nDataset --> " + args.dataset)
	status.write("\nModel --> " + args.model)
	status.write("\nCrop Height -->" + str(args.crop_height))
	status.write("\nCrop Width -->" + str(args.crop_width))
	status.write("\nNum Epochs -->" + str(args.num_epochs))
	status.write("\nBatch Size -->" + str(args.batch_size))
	status.close()

# Load the data
print("Loading the data ...")
# ["he_at_me", "le_at_me", "me_at_he", "me_at_le", "he", "le", "me"]
exposure_keys_train = ["he", "le", "me"]
exposure_keys_train_labels = ["hdr"]
exposure_keys_val = ["he", "le", "me"]
exposure_keys_val_labels = ["hdr"]

multiexposure_train_names = utils.prepare_data_multiexposure("%s/train_256"%(args.dataset), exposure_keys_train)
multiexposure_train_label_names = utils.prepare_data_multiexposure("%s/train_labels_256"%(args.dataset), exposure_keys_train_labels)
multiexposure_val_names = utils.prepare_data_multiexposure("%s/val"%(args.dataset), exposure_keys_val)
multiexposure_val_label_names = utils.prepare_data_multiexposure("%s/val_labels"%(args.dataset), exposure_keys_val_labels)
train_input_names_he, train_input_names_le, train_input_names_me = multiexposure_train_names[0], multiexposure_train_names[1], multiexposure_train_names[2]
train_output_names_hdr = multiexposure_train_label_names[0]
val_input_names_he, val_input_names_le, val_input_names_me = multiexposure_val_names[0], multiexposure_val_names[1], multiexposure_val_names[2]
val_output_names_hdr = multiexposure_val_label_names[0]



# Which validation images do we want
val_indices = []
num_vals = min(args.num_val_images, len(val_input_names_he))

# Set random seed to make sure models are validated on the same validation images.
# So you can compare the results of different models more intuitively.
random.seed(16)
val_indices=random.sample(range(0,len(val_input_names_he)),num_vals)
learning_rates = []
lr_decay_step = 1
small_loss_bin = []



train_step =0
val_step = 0

# Do the training here
for epoch in range(args.epoch_start_i, args.num_epochs):

	learning_rate = args.init_lr*(float)(args.lr_decay)**(float)(epoch)
	learning_rates.append(learning_rate)
	print("\nLearning rate for epoch # %04d = %f\n"%(epoch, learning_rate))
	if args.save_logs:
		log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
		log_file.write("\nLearning rate for epoch " + str(epoch) + " = " + str(learning_rate) + "\n")
		log_file.close()            
		status = open("status%d.txt"%(args.status_id),'a')                        
		status.write("\nLearning rate for epoch " + str(epoch) + " = " + str(learning_rate) + "\n")
		status.close()            

	current_losses = []
	current_losses_val = []
	cnt=0
	
	# Equivalent to shuffling
	id_list = np.random.permutation(len(train_input_names_he))
	num_iters = int(np.floor(len(id_list) / args.batch_size))
	st = time.time()
	epoch_st=time.time()
	


	for i in range(num_iters):
		input_image_le_batch = []
		input_image_me_batch = []
		input_image_he_batch = []

		output_image_batch = []
		
		# Collect a batch of images
		for j in range(args.batch_size):
			index = i*args.batch_size + j
			id = id_list[index]
			cv2_image_train_he = cv2.imread(train_input_names_he[id],-1)
			input_image_he = np.float32(cv2.cvtColor(cv2_image_train_he,cv2.COLOR_BGR2RGB)) / 65535.0
			input_image_he_gamma,_,_ = utils.ldr_to_hdr_train(input_image_he,train_input_names_he[id])
			input_image_he_c = np.concatenate([input_image_he,input_image_he_gamma],axis=2)


			cv2_image_train_me = cv2.imread(train_input_names_me[id],-1)
			input_image_me = np.float32(cv2.cvtColor(cv2_image_train_me,cv2.COLOR_BGR2RGB)) / 65535.0
			input_image_me_gamma,_,_ = utils.ldr_to_hdr_train(input_image_me,train_input_names_me[id])
			input_image_me_c = np.concatenate([input_image_me,input_image_me_gamma],axis=2)


			cv2_image_train_le = cv2.imread(train_input_names_le[id],-1)
			input_image_le = np.float32(cv2.cvtColor(cv2_image_train_le,cv2.COLOR_BGR2RGB)) / 65535.0
			input_image_le_gamma,_,_ = utils.ldr_to_hdr_train(input_image_le,train_input_names_le[id])
			input_image_le_c = np.concatenate([input_image_le,input_image_le_gamma],axis=2)


			output_image = cv2.cvtColor(cv2.imread(train_output_names_hdr[id],-1),cv2.COLOR_BGR2RGB)


			input_image_le_batch.append(np.expand_dims(input_image_le_c, axis=0))
			input_image_me_batch.append(np.expand_dims(input_image_me_c, axis=0))
			input_image_he_batch.append(np.expand_dims(input_image_he_c, axis=0))
			output_image_batch.append(np.expand_dims(output_image, axis=0))
		

		
		input_image_le_batch = np.squeeze(np.stack(input_image_le_batch, axis=1))
		input_image_me_batch = np.squeeze(np.stack(input_image_me_batch, axis=1))
		input_image_he_batch = np.squeeze(np.stack(input_image_he_batch, axis=1))
		output_image_batch =   np.squeeze(np.stack(output_image_batch, axis=1))
 

		
		
		train_writer.add_summary(sess.run(le_image_summ,feed_dict={le_image_pl:input_image_le_batch[...,:3]}),i)
		train_writer.add_summary(sess.run(me_image_summ,feed_dict={me_image_pl:input_image_me_batch[...,:3]}),i)
		train_writer.add_summary(sess.run(he_image_summ,feed_dict={he_image_pl:input_image_he_batch[...,:3]}),i)
		train_writer.add_summary(sess.run(gt_image_summ,feed_dict={gt_image_pl:output_image_batch[...,:3]}),i)

		
		# Do the training here
		_,current_loss=sess.run([opt,loss],feed_dict={input_exposure_stacks[0]:input_image_le_batch,input_exposure_stacks[1]:input_image_me_batch,input_exposure_stacks[2]:input_image_he_batch, gt_exposure_stack:output_image_batch, lr:learning_rate})
				

		current_losses.append(current_loss)
		small_loss_bin.append(current_loss)

		cnt = cnt + args.batch_size

		if cnt % args.log_interval == 0:
			small_loss_bin_mean = np.mean(small_loss_bin)
			string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f "%(epoch, cnt, small_loss_bin_mean, time.time()-st)
			small_loss_bin = []
			train_str = utils.LOG(string_print)
			print(train_str)
			if args.save_logs:
				log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
				log_file.write(train_str + "\n")
				log_file.close()
				status = open("status%d.txt"%(args.status_id),'a')
				status.write(train_str + "\n")
				status.close()
			st = time.time()

			summ = sess.run(train_loss_summary, feed_dict={train_loss_pl:np.mean(current_losses)})
			train_writer.add_summary(summ,train_step)
			train_step +=1

	mean_loss = np.mean(current_losses)
	avg_loss_per_epoch.append(mean_loss)

	
	

	# Create directories if needed
	if not os.path.isdir("%s/%s/%04d"%(try_name, "ckpts", epoch)):
		os.makedirs("%s/%s/%04d"%(try_name, "ckpts", epoch))

	# Save latest checkpoint to same file name
	print("Saving latest checkpoint")
	saver.save(sess, model_checkpoint_name)

	if val_indices != 0 and epoch % args.checkpoint_step == 0:
		print("Saving checkpoint for this epoch")
		saver.save(sess, "%s/%s/%04d/model.ckpt"%(try_name, "ckpts", epoch))

	print("Average Training loss = ", mean_loss)


	if args.save_logs:
		log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
		log_file.write("\nAverage Training loss = " + str(mean_loss)) 
		log_file.close()
		status = open("status%d.txt"%(args.status_id),'a')
		status.write("\nAverage Training loss = " + str(mean_loss))
		status.close()  



	if epoch % args.validation_step == 0:

		print("Performing validation")
		if not os.path.isdir("%s/%s/%04d"%(try_name, "val_Imgs", epoch)):
			os.makedirs("%s/%s/%04d"%(try_name, "val_Imgs", epoch))

		psnr_pre_list = []
		psnr_post_list = []
		val_idx_count = 0
		pred_time_list = []
		# Do the validation on a small set of validation images
		for ind in val_indices:
			print("\rRunning test image %d / %d"%(val_idx_count+1, len(val_input_names_he)))
			input_images = []
			
			cv2_img_he = cv2.imread(val_input_names_he[ind],-1)
			h,w = cv2_img_he.shape[:2]			
			input_image_he = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_he,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
			input_image_he_gamma,_,_ = utils.ldr_to_hdr_test(input_image_he,val_input_names_he[ind])
			input_image_he_c = np.concatenate([input_image_he,input_image_he_gamma],axis=3)


			cv2_img_me = cv2.imread(val_input_names_me[ind],-1)
			h,w = cv2_img_me.shape[:2]
			input_image_me = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_me,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
			input_image_me_gamma,_,_ = utils.ldr_to_hdr_test(input_image_me,val_input_names_me[ind])
			input_image_me_c = np.concatenate([input_image_me,input_image_me_gamma],axis=3)
			

			cv2_img_le = cv2.imread(val_input_names_le[ind],-1)
			h,w = cv2_img_le.shape[:2]
			input_image_le = np.expand_dims(np.float32(cv2.cvtColor(cv2_img_le,cv2.COLOR_BGR2RGB)),axis=0)/65535.0
			input_image_le_gamma,_,_ = utils.ldr_to_hdr_test(input_image_le,val_input_names_le[ind])
			input_image_le_c = np.concatenate([input_image_le,input_image_le_gamma],axis=3)
			

			cv2_img_hdr = cv2.imread(val_output_names_hdr[ind],-1)
			h,w = cv2_img_hdr.shape[:2]
			gt_hdr = cv2.cvtColor(cv2_img_hdr,cv2.COLOR_BGR2RGB)
			gt_hdr = np.expand_dims(np.float32(gt_hdr), axis=0)

			pred_st = time.time()
			output_image_pred, curr_val_loss = sess.run([network,loss],feed_dict={input_exposure_stacks[0]:input_image_le_c,input_exposure_stacks[1]:input_image_me_c,input_exposure_stacks[2]:input_image_he_c,gt_exposure_stack:gt_hdr})
			pred_et = time.time()
			pred_time_list.append(pred_et-pred_st)


			output_image = np.squeeze(output_image_pred)
			gt_hdr = np.squeeze(gt_hdr)
			h,w = output_image.shape[:2]

			output_image_cropped = output_image[args.crop_pixels_height:h-args.crop_pixels_height,:,:]
			gt_hdr_cropped = gt_hdr[args.crop_pixels_height:h-args.crop_pixels_height,:,:]


			current_pre_psnr = compute_psnr(output_image_cropped, gt_hdr_cropped)
			current_post_psnr = compute_psnr(log_tonemap_output(output_image_cropped), log_tonemap_output(gt_hdr_cropped))
			
			current_losses_val.append(curr_val_loss)
			psnr_pre_list.append(current_pre_psnr)
			psnr_post_list.append(current_post_psnr)

			file_name = utils.filepath_to_name(val_input_names_he[ind])

			radiance_writer("%s/%s/%04d/%s_pred.hdr"%(try_name, "val_Imgs", epoch, file_name),output_image)
			radiance_writer("%s/%s/%04d/%s_gt.hdr"%(try_name, "val_Imgs", epoch, file_name),gt_hdr)
			val_idx_count = val_idx_count+1
			mean_val_loss = np.mean(current_losses_val)

			merge_summ = tf.summary.merge([test_loss_summary,test_psnr_summary])
			merge_summ = sess.run(merge_summ, feed_dict={test_loss_pl:mean_val_loss,test_psnr_pl:np.mean(psnr_post_list)})
			test_writer.add_summary(merge_summ,val_step)
			val_step+=1

		

		mean_pre_psnr = np.mean(psnr_pre_list)
		mean_post_psnr = np.mean(psnr_post_list)
		mean_proc_time = np.mean(pred_time_list)

		print('val psnr pre list {}\n'.format(psnr_pre_list))
		print('val psnr post list {}\n'.format(psnr_post_list)) 

		print("Average Validation loss = %f"%(mean_val_loss))
		print("Average PRE-PSNR = %f"%(mean_pre_psnr))
		print("Average POST -PSNR = %f"%(mean_post_psnr))
		print('Average processing time = %f'%(mean_proc_time))
		
		

			

		if args.save_logs:
			log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
			log_file.write("\nAverage Validation loss = " + str(mean_val_loss)+"\n")
			log_file.write("Average PRE-PSNR = %f\n"%(mean_pre_psnr))
			log_file.write("Average POST-PSNR = %f\n"%(mean_post_psnr))
			log_file.write('Average processing time = %f\n'%(mean_proc_time))
			status = open("status%d.txt"%(args.status_id),'a') 
			status.write("\nAverage Validation loss = " + str(mean_val_loss)+"\n")
			status.write("Average PRE-PSNR = %f\n"%(mean_pre_psnr))
			status.write("Average POST -PSNR = %f\n"%(mean_post_psnr))
			status.write('Average processing time = %f\n'%(mean_proc_time))
			status.close()



	epoch_time=time.time()-epoch_st
	remain_time=epoch_time*(args.num_epochs-1-epoch)
	m, s = divmod(remain_time, 60)
	h, m = divmod(m, 60)
	if s!=0:
		train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
	else:
		train_time="Remaining training time : Training completed.\n"
	str_time = utils.LOG(train_time)
	print(str_time)
	if args.save_logs:
		log_file = open("%s/Logs_try%d.txt"%(try_name, args.nTry), 'a')
		log_file.write(str_time + "\n")
		log_file.close()
		status = open("status%d.txt"%(args.status_id),'a')
		status.write(str_time + "\n")
		status.close()	

	

sess.close()







from retrain_official import *

class TransferLearning:
	def __init__(self, batch_size, num_epochs, architecture, classes_list):
		self.num_epochs = num_epochs
		self.architecture = architecture
		self.batch_size = batch_size
		self.categories_dir = \
		{
			'train' : train_dir,
			'test' : test_dir,
			'dev' : dev_dir
		}
		self.classes_list = classes_list
		self.class_count = len(self.classes_list)

	def _prepare_data(self):
		# TODO: check
		self.images_path_dict, self.true_labels_dict = create_image_lists(self.categories_dir)

	def _create_model_info(self):
		# Prepare necessary directories that can be used during training
		prepare_file_system()
		# Gather information about the model architecture we'll be using.
		self.model_info = create_model_info(FLAGS.architecture)
		if not self.model_info:
			tf.logging.error('Did not recognize architecture flag')
			return -1

	def _create_cache_bottlenecks(self, sess):

		def run_bottleneck_on_image(sess, image_path, label_name):
			image_data = gfile.FastGFile(image_path, 'rb').read()
			# First decode the JPEG image, resize it, and rescale the pixel values.
			resized_input_values = sess.run(self.decoded_image_tensor,
											{self.jpeg_data_tensor: image_data})

			# Then run it through the recognition network.
			bottleneck_values = sess.run(self.bottleneck_tensor,
										 {self.resized_image_tensor: resized_input_values})
			bottleneck_values = np.squeeze(bottleneck_values)
			return bottleneck_values

		def create_bottleneck_file(bottleneck_path, image_path, label_name):
			tf.logging.info('Creating bottleneck at ' + bottleneck_path)
			if not gfile.Exists(image_path):
				tf.logging.fatal('File does not exist %s', image_path)

			try:
				bottleneck_values = run_bottleneck_on_image(sess, image_path, label_name)
			except Exception as e:
				raise RuntimeError('Error during processing file %s (%s)' % (image_path,
																			 str(e)))
			bottleneck_string = ','.join(str(x) for x in bottleneck_values)
			with open(bottleneck_path, 'w') as bottleneck_file:
				bottleneck_file.write(bottleneck_string)

		def get_bottleneck_path(image_path, label_name):
			base_name = os.path.basename(image_path)
			return os.path.join(FLAGS.bottleneck_dir, label_name, base_name) + '.txt'

		def get_or_create_bottleneck(sess, image_path, label_name):
			sub_dir_path = os.path.join(FLAGS.bottleneck_dir, label_name)
			ensure_dir_exists(sub_dir_path)
			bottleneck_path = get_bottleneck_path(image_path, label_name)
			if not os.path.exists(bottleneck_path):
				create_bottleneck_file(bottleneck_path, image_path, label_name)
			with open(bottleneck_path, 'r') as bottleneck_file:
				bottleneck_string = bottleneck_file.read()
			did_hit_error = False
			bottleneck_values = -1
			try:
				bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
			except ValueError:
				tf.logging.warning('Invalid float found, recreating bottleneck')
				did_hit_error = True
			if did_hit_error:
				create_bottleneck_file(bottleneck_path, image_path, label_name)
				with open(bottleneck_path, 'r') as bottleneck_file:
					bottleneck_string = bottleneck_file.read()
				bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

			assert bottleneck_values != -1
			return bottleneck_values

		how_many_bottlenecks = 0
		ensure_dir_exists(FLAGS.bottleneck_dir)
		self.bottlenecks_dict = defaultdict(list)
		self.one_hot_labels_list = defaultdict(list)
		for category, images_list_this_category in self.images_path_dict.items():
			for id, each_image_path in enumerate(images_list_this_category):
				true_label_this_image = self.true_labels_dict[category][id]
				bottleneck = get_or_create_bottleneck(sess, each_image_path, true_label_this_image)
				self.bottlenecks_dict[category].append(bottleneck)

				one_hot_label = np.zeros(len(self.classes_list), dtype=np.float32)
				label_index = self.classes_list.index(true_label_this_image)
				one_hot_label[label_index] = 1.0
				self.one_hot_labels_list[category].append(one_hot_label)

	def _create_input_tensors(self):
		self.graph, self.bottleneck_tensor, self.resized_image_tensor = (create_model_graph(self.model_info))
		self.jpeg_data_tensor, self.decoded_image_tensor = add_jpeg_decoding(
			self.model_info['input_width'], self.model_info['input_height'],
			self.model_info['input_depth'], self.model_info['input_mean'],
			self.model_info['input_std'])
		(self.distorted_jpeg_data_tensor,
		 self.distorted_image_tensor) = add_input_distortions(
			FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
			FLAGS.random_brightness, self.model_info['input_width'],
			self.model_info['input_height'], self.model_info['input_depth'],
			self.model_info['input_mean'], self.model_info['input_std'])

	def _create_new_layer(self):
		# Add the new layer that we'll be training.
		self.bottleneck_tensor_size=self.model_info['bottleneck_tensor_size']
		with tf.name_scope('input'):
			self.bottleneck_input = tf.placeholder_with_default(
				self.bottleneck_tensor,
				shape=[None, self.bottleneck_tensor_size],
				name='BottleneckInputPlaceholder')

			self.one_hot_labels = tf.placeholder(tf.float32, [None, self.class_count], name='One_hot_labels')
		# Organizing the following ops as `final_training_ops` so they're easier
		# to see in TensorBoard
		layer_name = 'final_training_ops'
		with tf.name_scope(layer_name):
			with tf.name_scope('weights'):
				initial_value = tf.truncated_normal(
					[self.bottleneck_tensor_size, self.class_count], stddev=0.001)

				self.layer_weights = tf.Variable(initial_value, name='final_weights')

				variable_summaries(self.layer_weights)
			with tf.name_scope('biases'):
				self.layer_biases = tf.Variable(tf.zeros([self.class_count]), name='final_biases')
				variable_summaries(self.layer_biases)
			with tf.name_scope('Wx_plus_b'):
				self.logits = tf.matmul(self.bottleneck_input, self.layer_weights) + self.layer_biases
				tf.summary.histogram('pre_activations', self.logits)

		final_tensor_name = FLAGS.final_tensor_name
		self.softmaxed_logits = tf.nn.softmax(self.logits, name=final_tensor_name)
		tf.summary.histogram('activations', self.softmaxed_logits)

		with tf.name_scope('cross_entropy'):
			cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
				labels=self.one_hot_labels, logits=self.logits)
			with tf.name_scope('total'):
				self.cross_entropy_mean = tf.reduce_mean(cross_entropy)
		tf.summary.scalar('cross_entropy', self.cross_entropy_mean)

		with tf.name_scope('train'):
			self.optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(self.cross_entropy_mean)

	def _create_evalation(self):
		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				self.predicted_labels = tf.argmax(self.softmaxed_logits, 1)
				correct_prediction = tf.equal(self.predicted_labels, tf.argmax(self.one_hot_labels, 1))

			with tf.name_scope('accuracy'):
				self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
			with tf.name_scope('precision'):
				self.precision = tf.metrics.precision(tf.argmax(self.one_hot_labels, 1), self.predicted_labels)

	def _create_summaries(self):
		tf.summary.scalar('accuracy', self.accuracy)
		tf.summary.scalar('precision', self.precision)
		self.summary_merged = tf.summary.merge_all()

def batch_generator(bottlenecks_list, labels_list, batch_size, shuffle=True):
	bottlenecks_list_tensor = tf.convert_to_tensor(bottlenecks_list)
	one_hot_labels_list_tensor = tf.convert_to_tensor(labels_list)
	min_after_dequeue = 10 * batch_size
	capacity = 20 * batch_size
	data_batch, label_batch = tf.train.shuffle_batch([bottlenecks_list_tensor, one_hot_labels_list_tensor],
													 batch_size=batch_size,
													 capacity=capacity,
													 min_after_dequeue=min_after_dequeue)
	return data_batch, label_batch

import math

def train_model(model):
	do_distort_images = should_distort_images(
		FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
		FLAGS.random_brightness)

	with tf.Session(graph=model.graph) as sess:
		if not do_distort_images:
			model._create_cache_bottlenecks(sess)

		# Merge all the summaries and write them out to the summaries_dir
		train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
		validation_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
		init = tf.global_variables_initializer()
		sess.run(init)

		num_batches_per_epoch = math.ceil(len(model.images_path_dict['train'])/ FLAGS.batch_size)
		train_data_batch, train_labels_batch = batch_generator(model.bottlenecks_dict['train'],
													model.one_hot_labels_list['train'],
													FLAGS.batch_size,
													shuffle=True)

		dev_data_batch, dev_labels_batch = batch_generator(model.bottlenecks_dict['dev'],
													model.one_hot_labels_list['dev'],
													FLAGS.batch_size,
													shuffle=True)

		num_steps = num_batches_per_epoch * FLAGS.num_epochs
		for step in xrange(num_steps):
			current_epoch = step / num_batches_per_epoch + 1
			#at the first epoch
			if step % num_batches_per_epoch == 0:
				accuracy, loss = sess.run([model.accuracy, model.cross_entropy_mean],
										feed_dict={model.bottleneck_input: train_data_batch,
												   model.one_hot_labels: train_labels_batch})
				logging.info('Epoch %s', current_epoch)
				logging.info('Current Accuracy: %s', accuracy)
				logging.info('Current Loss: %s', loss)



def main(_):
	# Needed to make sure the logging output is visible.
	# See https://github.com/tensorflow/tensorflow/issues/3047
	tf.logging.set_verbosity(tf.logging.INFO)
	classes_list = ['Car_accident', 'Death_tragedy', 'Safe']
	model=TransferLearning(FLAGS.batch_size, FLAGS.num_epochs, FLAGS.architecture, classes_list)
	train_model(model)
	evaluate_model(model)



		# Set up all our weights to their initial default values.




		# Run the training for as many cycles as requested on the command line.
		for step in xrange(num_batches_per_epoch * FLAGS.num_epochs):
			current_epoch = step / num_batches_per_epoch + 1
			# Get a batch of input bottleneck values, either calculated fresh every
			# time with distortions applied, or from the cache stored on disk.
			if do_distort_images:
				(train_bottlenecks, train_ground_truth) = get_random_distorted_bottlenecks(
					sess, image_lists, FLAGS.train_batch_size, 'train',
					FLAGS.train_dir, distorted_jpeg_data_tensor,
					distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
			else:
				(train_bottlenecks, train_ground_truth, _) = get_random_cached_bottlenecks(
					sess, image_lists, FLAGS.train_batch_size, 'train',
					FLAGS.bottleneck_dir, FLAGS.train_dir, jpeg_data_tensor,
					decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
					FLAGS.architecture)

			# Every so often, print out how well the graph is training.
			# At the start of every epoch, show the vital information:
			if step % num_batches_per_epoch == 0:
				train_accuracy, cross_entropy_value = sess.run([evaluation_step, cross_entropy],
					feed_dict={bottleneck_input: train_bottlenecks,
							   ground_truth_input: train_ground_truth})

				tf.logging.info('%s: Epoch %d: Train accuracy = %.1f%%' %
								(datetime.now(), current_epoch, train_accuracy * 100))
				tf.logging.info('%s: Epoch %d: Cross entropy = %f' %
								(datetime.now(), current_epoch, cross_entropy_value))

				validation_bottlenecks, validation_ground_truth, _ = (
					get_random_cached_bottlenecks(
						sess, image_lists, FLAGS.validation_batch_size, 'dev',
						FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
						decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
						FLAGS.architecture))

				# Run a validation step and capture training summaries for TensorBoard
				# with the `merged` op.
				validation_summary, validation_accuracy = sess.run(
					[merged, evaluation_step],
					feed_dict={bottleneck_input: validation_bottlenecks,
							   ground_truth_input: validation_ground_truth})

				validation_writer.add_summary(validation_summary, current_epoch)
				tf.logging.info('%s: Epoch %d: Validation accuracy = %.1f%% (N=%d)' %
								(datetime.now(), current_epoch, validation_accuracy * 100,
								 len(validation_bottlenecks)))

			# Feed the bottlenecks and ground truth into the graph, and run a training
			# step. Capture training summaries for TensorBoard with the `merged` op.
			train_summary, _ = sess.run([merged, train_step],
								feed_dict={bottleneck_input: train_bottlenecks,
										   ground_truth_input: train_ground_truth})
			train_writer.add_summary(train_summary, step)

		# We've completed all our training, so run a final test evaluation on
		# some new images we haven't used before.
		test_bottlenecks, test_ground_truth, test_filenames = (
			get_random_cached_bottlenecks(
				sess, image_lists, FLAGS.test_batch_size, 'test',
				FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
				decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
				FLAGS.architecture))

		print(len(test_bottlenecks))
		test_accuracy, predictions = sess.run(
			[evaluation_step, prediction],
			feed_dict={bottleneck_input: test_bottlenecks,
						 ground_truth_input: test_ground_truth})
		# tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
		#								 (test_accuracy * 100, len(test_bottlenecks)))
		print('Final test accuracy = %.1f%% (N=%d)' %
				(test_accuracy * 100, len(test_bottlenecks)))

		if FLAGS.print_misclassified_test_images:
			tf.logging.info('=== MISCLASSIFIED TEST IMAGES ===')
			for i, test_filename in enumerate(test_filenames):
				if predictions[i] != test_ground_truth[i].argmax():
					print('%70s	%s' %
							(test_filename,
							 list(image_lists.keys())[predictions[i]]))

		# Write out the trained graph and labels with the weights stored as
		# constants.
		save_graph_to_file(sess, graph, FLAGS.output_graph)
		with gfile.FastGFile(FLAGS.output_labels, 'w') as f:
			f.write('\n'.join(image_lists.keys()) + '\n')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# parser.add_argument(
	#		 '--image_dir',
	#		 type=str,
	#		 default='',
	#		 help='Path to folders of labeled images.'
	# )
	parser.add_argument(
		'--train_dir',
		type=str,
		default=''
	)
	parser.add_argument(
		'--dev_dir',
		type=str,
		default='',
	)
	parser.add_argument(
		'--test_dir',
		type=str,
		default='',
	)
	parser.add_argument(
		'--output_graph',
		type=str,
		default='./output_dir/output_graph.pb',
		help='Where to save the trained graph.'
	)
	parser.add_argument(
		'--intermediate_output_graphs_dir',
		type=str,
		default='./output_dir/intermediate_graph/',
		help='Where to save the intermediate graphs.'
	)
	parser.add_argument(
		'--intermediate_store_frequency',
		type=int,
		default=0,
		help="""\
		 How many steps to store intermediate graph. If "0" then will not
		 store.\
		"""
	)
	parser.add_argument(
		'--output_labels',
		type=str,
		default='./output_dir/output_labels.txt',
		help='Where to save the trained graph\'s labels.'
	)
	parser.add_argument(
		'--summaries_dir',
		type=str,
		default='./tmp/retrain_logs',
		help='Where to save summary logs for TensorBoard.'
	)
	parser.add_argument(
		'--num_epochs',
		type=int,
		default=4000,
		help='How many training steps to run before ending.'
	)
	parser.add_argument(
		'--learning_rate',
		type=float,
		default=0.01,
		help='How large a learning rate to use when training.'
	)
	parser.add_argument(
		'--testing_percentage',
		type=int,
		default=10,
		help='What percentage of images to use as a test set.'
	)
	parser.add_argument(
		'--validation_percentage',
		type=int,
		default=5,
		help='What percentage of images to use as a validation set.'
	)
	parser.add_argument(
		'--eval_step_interval',
		type=int,
		default=50,
		help='How often to evaluate the training results.'
	)
	parser.add_argument(
		'--batch_size',
		type=int,
		default=20,
		help='How many images to train on at a time.'
	)
	parser.add_argument(
		'--test_batch_size',
		type=int,
		default=-1,
		help="""\
		How many images to test on. This test set is only used once, to evaluate
		the final accuracy of the model after training completes.
		A value of -1 causes the entire test set to be used, which leads to more
		stable results across runs.\
		"""
	)
	parser.add_argument(
		'--validation_batch_size',
		type=int,
		default=10,
		help="""\
		How many images to use in an evaluation batch. This validation set is
		used much more often than the test set, and is an early indicator of how
		accurate the model is during training.
		A value of -1 causes the entire validation set to be used, which leads to
		more stable results across training iterations, but may be slower on large
		training sets.\
		"""
	)
	parser.add_argument(
		'--print_misclassified_test_images',
		default=False,
		help="""\
		Whether to print out a list of all misclassified test images.\
		""",
		action='store_true'
	)
	parser.add_argument(
		'--model_dir',
		type=str,
		default='./inception',
		help="""\
		Path to classify_image_graph_def.pb,
		imagenet_synset_to_human_label_map.txt, and
		imagenet_2012_challenge_label_map_proto.pbtxt.\
		"""
	)
	parser.add_argument(
		'--bottleneck_dir',
		type=str,
		default='./tmp/bottleneck',
		help='Path to cache bottleneck layer values as files.'
	)
	parser.add_argument(
		'--final_tensor_name',
		type=str,
		default='final_result',
		help="""\
		The name of the output classification layer in the retrained graph.\
		"""
	)
	parser.add_argument(
		'--flip_left_right',
		default=False,
		help="""\
		Whether to randomly flip half of the training images horizontally.\
		""",
		action='store_true'
	)
	parser.add_argument(
		'--random_crop',
		type=int,
		default=0,
		help="""\
		A percentage determining how much of a margin to randomly crop off the
		training images.\
		"""
	)
	parser.add_argument(
		'--random_scale',
		type=int,
		default=0,
		help="""\
		A percentage determining how much to randomly scale up the size of the
		training images by.\
		"""
	)
	parser.add_argument(
		'--random_brightness',
		type=int,
		default=0,
		help="""\
		A percentage determining how much to randomly multiply the training image
		input pixels up or down by.\
		"""
	)
	parser.add_argument(
		'--architecture',
		type=str,
		default='inception_v3',
		help="""\
		Which model architecture to use. 'inception_v3' is the most accurate, but
		also the slowest. For faster or smaller models, chose a MobileNet with the
		form 'mobilenet_<parameter size>_<input_size>[_quantized]'. For example,
		'mobilenet_1.0_224' will pick a model that is 17 MB in size and takes 224
		pixel input images, while 'mobilenet_0.25_128_quantized' will choose a much
		less accurate, but smaller and faster network that's 920 KB on disk and
		takes 128x128 images. See https://research.googleblog.com/2017/06/mobilenets-open-source-models-for.html
		for more information on Mobilenet.\
		""")
	FLAGS, unparsed = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


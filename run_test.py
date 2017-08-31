import sys
import tensorflow as tf
import cPickle as pkl
from os import listdir
from os.path import isfile, join
# change this as you see fit
source_dir = sys.argv[1]

#Threshold
THRESHOLD = sys.argv[2]

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("./output_dir/labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./output_dir/graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

file_list = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
writter = open('result_new.txt', 'wb')

NUM_LABELS = 3 
print 'PRINT LABELS!!! '
for id in range(NUM_LABELS):
	print label_lines[id]

with tf.Session() as sess:
    # Feed the image_data as input to the graph and get first prediction
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    for true_filename in file_list:
        #print (true_filename) 
	image_path = join(source_dir, true_filename)

        # Read in the image_data
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()

        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        # top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
        #
        # for node_id in top_k:
        #     human_string = label_lines[node_id]
        #     score = predictions[0][node_id]
        #     print('%s (score = %.5f)' % (human_string, score))
	
	if predictions[0][2] < 0.5:
        	writter.write(str(predictions[0][0]) + ' ' + str(predictions[0][1]) + ' ' + str(predictions[0][2]) + ' ' + true_filename + '\n')
	#if predictions[0][0] < 0.5:
	#	writter.write(true_filename + '\n') 

